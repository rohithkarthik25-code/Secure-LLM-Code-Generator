import networkx as nx
import matplotlib.pyplot as plt
import os
from capextract.core.models import CapabilityGraphIR, NodeType

def visualize_graph(graph: CapabilityGraphIR, output_path: str = "capability_graph.png"):
    """
    Generates a visual representation of the Capability Graph using NetworkX and Matplotlib.
    Saves the output to the specified path.
    """
    G = nx.DiGraph()
    
    # Track node types for coloring
    color_map = []
    
    # Add Nodes
    for node in graph.nodes.values():
        # Determine color and shape based on type
        if node.node_type == NodeType.PRIMITIVE:
            color = "lightblue"
        elif node.node_type == NodeType.FUNCTIONAL:
            color = "lightgreen"
        elif node.node_type == NodeType.RISK:
            color = "lightcoral"
        elif node.node_type == NodeType.INTENT:
            color = "orange"
        else:
            color = "gray"
            
        # Label string includes confidence if applicable
        label = f"{node.label}\n({node.confidence:.2f})"
        
        G.add_node(node.id, label=label, color=color, type=node.node_type)
        
    # Add Edges
    for edge in graph.edges:
        G.add_edge(edge.src, edge.dst, weight=edge.weight, label=f"{edge.edge_type.value}\n({edge.weight:.2f})")
        
    # Setup plot
    plt.figure(figsize=(12, 10))
    
    # Try different layouts
    # Kamada-kawai works well for these types of graphs
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, k=0.5)
        
    # Extract colors in the order of G.nodes
    node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
    node_labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
    edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=1.5, alpha=0.6)
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Capability Graph (Intent, Primitive, Functional, Risk)")
    plt.axis('off')
    
    # Save to file
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Visualizer] Graph saved to {output_path}")
