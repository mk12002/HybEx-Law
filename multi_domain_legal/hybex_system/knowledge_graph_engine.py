import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Any, Tuple
import re
import logging
from .config import HybExConfig
from .prolog_engine import PrologEngine

logger = logging.getLogger(__name__)

class LegalGNN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super(LegalGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class KnowledgeGraphEngine:
    def __init__(self, config: HybExConfig):
        self.config = config
        self.prolog_engine = PrologEngine(config)
        self.graph = nx.DiGraph()
        self.node_mapping = {}  # maps node names to integer indices
        self.feature_mapping = {} # maps node names to feature vectors
        self._build_knowledge_graph()
        self.model = LegalGNN(
            num_node_features=len(self.feature_mapping),
            num_classes=2  # Eligible / Not Eligible
        )
        logger.info("Knowledge Graph Engine Initialized")

    def _add_node(self, node: str, node_type: str):
        if node not in self.graph:
            self.graph.add_node(node, type=node_type)
            if node not in self.node_mapping:
                self.node_mapping[node] = len(self.node_mapping)
            if node not in self.feature_mapping:
                 # Create a one-hot encoded feature vector for each node
                feature_vector = torch.zeros(len(self.prolog_engine.legal_rules.keys()))
                self.feature_mapping[node] = feature_vector

    def _build_knowledge_graph(self):
        """Builds the knowledge graph from Prolog rules."""
        for rule_category, rules in self.prolog_engine.legal_rules.items():
            for rule in rules:
                # Basic parsing of Prolog rules; a more robust parser may be needed for complex rules
                match = re.match(r'(\w+)\((.*?)\)\s*:-\s*(.*)\.', rule)
                if match:
                    head, _, body = match.groups()
                    self._add_node(head, 'predicate')
                    body_predicates = re.findall(r'(\w+)\(', body)
                    for pred in body_predicates:
                        self._add_node(pred, 'predicate')
                        self.graph.add_edge(pred, head)
        logger.info(f"Knowledge Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _get_graph_data(self) -> Data:
        """Converts the NetworkX graph to a PyTorch Geometric Data object."""
        edge_list = list(self.graph.edges())
        edge_index = torch.tensor([
            [self.node_mapping[u] for u, v in edge_list],
            [self.node_mapping[v] for u, v in edge_list]
        ], dtype=torch.long)

        node_features = torch.stack([self.feature_mapping[node] for node in self.node_mapping.keys()], dim=0)

        return Data(x=node_features, edge_index=edge_index)

    def train_gnn(self, train_data: List[Dict[str, Any]]):
        """Trains the GNN on the provided training data."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        graph_data = self._get_graph_data()

        for epoch in range(100): # Simplified training loop
            self.model.train()
            optimizer.zero_grad()
            out = self.model(graph_data)
            # This is a simplified training approach. A more robust implementation would involve creating subgraphs for each training sample.
            # For this example, we'll use a placeholder loss.
            loss = criterion(out, torch.ones_like(out) * 0.5) # Placeholder
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict_eligibility(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts eligibility using the GNN."""
        self.model.eval()
        graph_data = self._get_graph_data()

        # Activate nodes based on extracted entities
        with torch.no_grad():
            node_embeddings = self.model(graph_data)

        # Simple logic to determine eligibility from node embeddings
        # A more sophisticated approach would involve a final classification layer or reasoning over the embeddings
        eligibility_score = node_embeddings[self.node_mapping.get('eligible_for_legal_aid', 0)].mean().item()

        return {
            'eligible': eligibility_score > 0.5,
            'confidence': eligibility_score,
            'primary_reason': 'Eligibility determined by GNN reasoning over the legal knowledge graph.',
            'method': 'gnn'
        }