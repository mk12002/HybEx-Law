import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import networkx as nx
from typing import Dict, List, Any, Tuple
import re
import logging
from .config import HybExConfig
from .prolog_engine import PrologEngine

logger = logging.getLogger(__name__)

class LegalGAT(torch.nn.Module):
    """
    A Graph Attention Network (GAT) for legal reasoning.
    This architecture is more powerful than GCN as it allows nodes to
    weigh the importance of their neighbors.
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int = 64, heads: int = 4):
        super(LegalGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.readout = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer with dropout and ELU activation
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer with dropout
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # The output `x` now contains the learned embeddings for each node.
        # A readout layer will be used in the engine to get a graph-level prediction.
        return x

class KnowledgeGraphEngine:
    """
    A sophisticated engine for building, training, and reasoning with a legal knowledge graph.
    """
    def __init__(self, config: HybExConfig):
        self.config = config
        self.prolog_engine = PrologEngine(config)
        self.graph = nx.DiGraph()
        
        # Node and feature mappings
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.relations = set()
        
        # Build the graph from Prolog rules
        self._build_knowledge_graph()
        
        # Initialize the GAT model
        self.model = LegalGAT(
            num_node_features=len(self.node_to_idx),  # One-hot encoding for node types
            num_classes=1,  # Binary classification: eligible or not
            hidden_channels=128,
            heads=8
        )
        logger.info(f"Knowledge Graph Engine Initialized with GAT model. Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _add_node(self, node: str, node_type: str):
        """Adds a node to the graph and updates mappings."""
        if node not in self.node_to_idx:
            idx = len(self.node_to_idx)
            self.node_to_idx[node] = idx
            self.idx_to_node[idx] = node
            self.graph.add_node(node, type=node_type, id=idx)

    def _build_knowledge_graph(self):
        """
        Builds a detailed knowledge graph from Prolog rules using a more robust parsing approach.
        Nodes can be predicates, entities, or literal values.
        """
        # A more robust regex to capture head, arguments, and body components
        rule_pattern = re.compile(r'([\w_]+)\((.*?)\)\s*:-\s*(.*)\.')

        for rules in self.prolog_engine.legal_rules.values():
            for rule in rules:
                rule = rule.strip()
                if not rule or rule.startswith('%'):
                    continue

                match = rule_pattern.match(rule)
                if match:
                    head_predicate, head_args_str, body_str = match.groups()
                    
                    # Add head predicate as a node
                    self._add_node(head_predicate, 'predicate')
                    
                    # Process body predicates
                    # This regex finds all predicate(... , ...) structures in the body
                    body_predicates = re.findall(r'([\w_]+)\(', body_str)
                    for body_pred in body_predicates:
                        self._add_node(body_pred, 'predicate')
                        # Add a directed edge from the body predicate to the head predicate
                        # This represents the logical dependency: body -> head
                        self.graph.add_edge(body_pred, head_predicate, relation='implies')
                        self.relations.add('implies')

        logger.info(f"Knowledge Graph built with {len(self.node_to_idx)} unique nodes (predicates).")

    def _create_case_subgraph(self, entities: Dict[str, Any]) -> Data:
        """
        Creates a PyTorch Geometric Data object for a specific case,
        "activating" nodes corresponding to the case facts.
        """
        # Start with the full graph structure
        edge_index = torch.tensor(list(self.graph.edges()), dtype=torch.long).t().contiguous()
        node_indices = [self.node_to_idx[u] for u, v in self.graph.edges()] + [self.node_to_idx[v] for u, v in self.graph.edges()]
        node_indices = torch.tensor(node_indices, dtype=torch.long)
        edge_index = torch.tensor([[self.node_to_idx[u] for u,v in self.graph.edges()],
                                   [self.node_to_idx[v] for u,v in self.graph.edges()]], dtype=torch.long)


        # Create base features (one-hot encoding of node indices)
        num_nodes = len(self.node_to_idx)
        x = F.one_hot(torch.arange(num_nodes), num_classes=num_nodes).float()

        # "Activate" nodes based on entities. For simplicity, we can just mark them.
        # A more advanced approach would be to have different features for different entity types.
        for key, value in entities.items():
            if key in self.node_to_idx:
                idx = self.node_to_idx[key]
                # Enhance the feature of the activated node
                x[idx] *= 2.0 

        return Data(x=x, edge_index=edge_index)

    def train_gnn(self, train_data: List[Dict[str, Any]], epochs: int = 50):
        """
        Trains the GNN on the provided training data, creating subgraphs for each sample.
        """
        logger.info(f"Starting GNN training for {epochs} epochs.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Create a dataset of case subgraphs
        graph_dataset = []
        for sample in train_data:
            entities = sample.get('extracted_entities', {})
            label = torch.tensor([float(sample.get('expected_eligibility', 0.0))], dtype=torch.float)
            graph_data = self._create_case_subgraph(entities)
            graph_data.y = label
            graph_dataset.append(graph_data)
            
        train_loader = DataLoader(graph_dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Get node embeddings from the model
                node_embeddings = self.model(batch)
                
                # Global mean pooling for graph-level representation
                graph_embedding = torch.mean(node_embeddings, dim=0)
                
                # Final prediction from the graph embedding
                out = self.model.readout(graph_embedding).squeeze(-1)
                
                loss = criterion(out, batch.y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:03d}, Loss: {avg_loss:.4f}")

    def predict_eligibility(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts eligibility for a given set of case entities using the trained GNN.
        """
        self.model.eval()
        
        # Create the case-specific subgraph
        case_graph_data = self._create_case_subgraph(entities)
        
        with torch.no_grad():
            # Get the learned embeddings for all nodes in the context of this case
            node_embeddings = self.model(case_graph_data)
            
            # Use global mean pooling to get a single vector representing the whole graph's state
            graph_embedding = torch.mean(node_embeddings, dim=0)
            
            # Pass the graph embedding through the final readout layer to get the prediction
            logit = self.model.readout(graph_embedding)
            
            # Apply sigmoid to get a probability
            eligibility_prob = torch.sigmoid(logit).item()

        return {
            'eligible': eligibility_prob > 0.5,
            'confidence': eligibility_prob,
            'primary_reason': 'Eligibility determined by GAT reasoning over the legal knowledge graph.',
            'method': 'gnn'
        }