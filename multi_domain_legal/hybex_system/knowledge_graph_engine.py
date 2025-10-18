import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import networkx as nx
from typing import Dict, List, Any, Tuple
import re
import logging
from sklearn.metrics import f1_score  # ✅ Moved to top for efficiency
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
        
        # Build the initial graph from Prolog rules
        self._build_knowledge_graph()
        
        # Model will be initialized after we know the feature dimension
        self.model = None
        self.feature_dim = None
        
        logger.info(f"Knowledge Graph Engine Initialized. Graph has {self.graph.number_of_nodes()} predicate nodes and {self.graph.number_of_edges()} edges.")

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
        rule_pattern = re.compile(r'([\w_]+)\(.*\)\s*:-\s*(.*)\.')

        for rules in self.prolog_engine.legal_rules.values():
            for rule in rules:
                rule = rule.strip()
                if not rule or rule.startswith('%'):
                    continue

                match = rule_pattern.match(rule)
                if match:
                    head_predicate, body_str = match.groups()
                    
                    # Add head predicate as a node
                    self._add_node(head_predicate, 'predicate')
                    
                    # Process body predicates
                    # This regex finds all predicate(... , ...) structures in the body
                    body_predicates = re.findall(r'(\w+)\(', body_str)
                    for body_pred in body_predicates:
                        self._add_node(body_pred, 'predicate')
                        # Add a directed edge from the body predicate to the head predicate
                        # This represents the logical dependency: body -> head
                        self.graph.add_edge(body_pred, head_predicate, relation='implies')
                        self.relations.add('implies')

        logger.info(f"Knowledge Graph built with {len(self.node_to_idx)} unique nodes (predicates).")
    
    def populate_with_training_cases(self, train_samples: List[Dict[str, Any]]):
        """
        Populates the knowledge graph with entity nodes from training cases.
        This creates a richer graph structure with both predicates and case-specific entities.
        """
        logger.info(f"Populating knowledge graph with {len(train_samples)} training cases...")
        
        entity_keys_added = set()
        
        for sample in train_samples:
            entities = sample.get('extracted_entities', {})
            
            if not entities:
                continue
            
            # Add entity nodes and connect them to relevant predicates
            for entity_key, entity_value in entities.items():
                if entity_value is None:
                    continue
                
                # Normalize entity key to match predicate names
                entity_node_name = entity_key  # e.g., 'annual_income', 'social_category'
                
                # Add entity type node if not already present
                if entity_node_name not in self.node_to_idx:
                    self._add_node(entity_node_name, 'entity_type')
                    entity_keys_added.add(entity_node_name)
                    
                    # Connect entity types to relevant predicates
                    # This creates the structural links in the graph
                    if 'income' in entity_key.lower():
                        for pred in ['eligible_for_legal_aid', 'income_eligible']:
                            if pred in self.node_to_idx:
                                self.graph.add_edge(entity_node_name, pred, relation='influences')
                                self.relations.add('influences')
                    
                    if 'category' in entity_key.lower() or 'social' in entity_key.lower():
                        for pred in ['eligible_for_legal_aid', 'categorically_eligible']:
                            if pred in self.node_to_idx:
                                self.graph.add_edge(entity_node_name, pred, relation='influences')
                                self.relations.add('influences')
                    
                    if 'case_type' in entity_key.lower():
                        for pred in ['eligible_for_legal_aid', 'legal_aid_applicable']:
                            if pred in self.node_to_idx:
                                self.graph.add_edge(entity_node_name, pred, relation='influences')
                                self.relations.add('influences')
        
        logger.info(f"✅ Knowledge graph populated. Total nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        logger.info(f"   - Predicate nodes from rules: {len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'predicate'])}")
        logger.info(f"   - Entity type nodes from cases: {len(entity_keys_added)}")
    
    def _create_case_graph(self, entities: Dict[str, Any]) -> Data:
        """
        Creates a PyTorch Geometric Data object for a specific case.
        Now uses entity-aware feature vectors.
        """
        if not self.graph.nodes():
            return Data(x=torch.empty((0, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
        
        num_nodes = len(self.node_to_idx)
        
        # Create edge index from graph structure
        edge_list = list(self.graph.edges())
        if not edge_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([
                [self.node_to_idx[u] for u, v in edge_list],
                [self.node_to_idx[v] for u, v in edge_list]
            ], dtype=torch.long)
        
        # Create node features: base one-hot + entity activation
        x = F.one_hot(torch.arange(num_nodes), num_classes=num_nodes).float()
        
        # Activate nodes corresponding to case entities with learned weights
        for entity_key, entity_value in entities.items():
            if entity_value is not None and entity_key in self.node_to_idx:
                idx = self.node_to_idx[entity_key]
                
                # Use entity value to create richer features
                if isinstance(entity_value, (int, float)):
                    # Normalize numeric values
                    normalized_value = min(float(entity_value) / 1000000.0, 10.0)  # Scale income
                    x[idx] = x[idx] * (1.0 + normalized_value)
                else:
                    # For categorical values, use stronger activation
                    x[idx] = x[idx] * 3.0
        
        return Data(x=x, edge_index=edge_index)

    def train_gnn(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]] = None, epochs: int = 100):
        """
        Trains the GNN on the provided training data.
        """
        # First, populate graph with training case entities
        self.populate_with_training_cases(train_data)
        
        # Check if graph has nodes
        if self.graph.number_of_nodes() == 0:
            logger.error("Knowledge graph is empty after population. Cannot train GNN.")
            return
        
        # Initialize the model now that we know feature dimensions
        self.feature_dim = len(self.node_to_idx)
        self.model = LegalGAT(
            num_node_features=self.feature_dim,
            num_classes=1,  # Binary classification
            hidden_channels=128,
            heads=8
        )
        
        logger.info(f"Starting GNN training for {epochs} epochs on {len(train_data)} samples.")
        logger.info(f"Feature dimension: {self.feature_dim}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create dataset
        train_dataset = []
        for sample in train_data:
            entities = sample.get('extracted_entities', {})
            label = float(sample.get('expected_eligibility', 0))
            
            graph_data = self._create_case_graph(entities)
            graph_data.y = torch.tensor([label], dtype=torch.float)
            train_dataset.append(graph_data)
        
        # Create validation dataset if provided
        val_dataset = []
        if val_data:
            for sample in val_data:
                entities = sample.get('extracted_entities', {})
                label = float(sample.get('expected_eligibility', 0))
                
                graph_data = self._create_case_graph(entities)
                graph_data.y = torch.tensor([label], dtype=torch.float)
                val_dataset.append(graph_data)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) if val_dataset else None
        
        best_val_f1 = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            train_preds, train_labels = [], []
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Get node embeddings
                node_embeddings = self.model(batch)
                
                # Global mean pooling for graph-level representation
                # Handle batched graphs properly
                batch_size = batch.y.size(0)
                graph_embeddings = []
                
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    graph_emb = node_embeddings[mask].mean(dim=0)
                    graph_embeddings.append(graph_emb)
                
                graph_embeddings = torch.stack(graph_embeddings)
                
                # Prediction
                out = self.model.readout(graph_embeddings).squeeze(-1)
                loss = criterion(out, batch.y.squeeze())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Track predictions
                preds = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(batch.y.cpu().numpy())
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Calculate train F1 (f1_score imported at top of file)
            train_f1 = f1_score(train_labels, train_preds, average='binary', zero_division=0)
            
            # Validation
            val_f1 = 0.0
            if val_loader:
                self.model.eval()
                val_preds, val_labels = [], []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        node_embeddings = self.model(batch)
                        
                        batch_size = batch.y.size(0)
                        graph_embeddings = []
                        
                        for i in range(batch_size):
                            mask = (batch.batch == i)
                            graph_emb = node_embeddings[mask].mean(dim=0)
                            graph_embeddings.append(graph_emb)
                        
                        graph_embeddings = torch.stack(graph_embeddings)
                        out = self.model.readout(graph_embeddings).squeeze(-1)
                        
                        preds = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(batch.y.cpu().numpy())
                
                val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0:
                if val_loader:
                    logger.info(f"Epoch {epoch+1:03d}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1:03d}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            
            # Early stopping check
            if val_loader and patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Best Val F1: {best_val_f1:.4f}")
                self.model.load_state_dict(self.best_model_state)
                break
        
        logger.info(f"✅ GNN training completed. Best Val F1: {best_val_f1:.4f}")

    def predict_eligibility(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts eligibility for a given set of case entities using the trained GNN.
        """
        if self.model is None:
            return {
                'eligible': False,
                'confidence': 0.0,
                'primary_reason': 'GNN model not trained yet.',
                'method': 'gnn_not_trained'
            }
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Create the case-specific graph
        case_graph_data = self._create_case_graph(entities)
        
        if case_graph_data.num_nodes == 0:
            return {
                'eligible': False,
                'confidence': 0.0,
                'primary_reason': 'Knowledge graph is empty.',
                'method': 'gnn_fallback'
            }
        
        case_graph_data = case_graph_data.to(device)
        
        with torch.no_grad():
            # Get node embeddings from GAT layers
            node_embeddings = self.model(case_graph_data)
            
            # Global mean pooling to get graph-level representation
            # For single graph (non-batched), just take mean across all nodes
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
            
            # Pass through readout layer to get prediction logit
            logit = self.model.readout(graph_embedding)
            
            # Apply sigmoid to get probability
            eligibility_prob = torch.sigmoid(logit).squeeze().item()

        return {
            'eligible': eligibility_prob > 0.5,
            'confidence': eligibility_prob,
            'primary_reason': f'GNN-based reasoning (confidence: {eligibility_prob:.2%})',
            'method': 'gnn'
        }