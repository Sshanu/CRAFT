import networkx as nx
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np

class PromptExpansionGraph:
    def __init__(self, logger):
        self.nodes = {}
        self.edges = []
        self.logger = logger

    def add_node(self, prompt):
        """
        Add or update a node represented by a PromptTree instance. If the node has a parent_id,
        automatically add an edge from the parent to this node.

        Args:
            prompt (PromptTree): The prompt object to add.
        """
        self.nodes[prompt.id] = prompt
        # Automatically add an edge if a parent is specified.
        if prompt.parent_id is not None:
            self.add_edge(prompt.parent_id, prompt.id)

    def add_edge(self, parent_id: str, child_id: str):
        """
        Add an edge from the parent prompt to the child prompt.
        """
        self.edges.append((parent_id, child_id))

    def update_node(self, prompt):
        """
        Update an existing node's data. If the node doesn't exist, it is added.
        """
        self.add_node(prompt)

    def update_nodes(self, prompts):
        """
        Update or add a list of PromptTree nodes to the graph.
        
        For each prompt in the list, this method calls update_node, which will add the node 
        if it does not exist or update it (and add the corresponding edge) if it does.
        
        Args:
            prompts (list): A list of PromptTree instances to update in the graph.
        """
        for prompt in prompts:
            self.update_node(prompt)

    def visualize(self, filename):
        """
        Visualize the expansion graph interactively using Plotly.
        Nodes are colored based on their round_created value and annotated with their id and test score.
        Directed edges are represented by arrow annotations.
        The graph is saved as an interactive HTML file, and the graph data is saved as JSON.
        """
        # Create a directed graph.
        try:
            G = nx.DiGraph()
        
            # Add nodes and create hover text labels.
            node_hover_labels = {}
            node_text = {}  # Text to display over each node.
            for id, prompt_obj in self.nodes.items():
                data = prompt_obj.get_info_dict()
                G.add_node(id, **data)
                round_val = data.get("round_created", 0)
                # Prepare a detailed hover label.
                node_hover_labels[id] = (
                    f"Id: {data.get('id', 'N/A')}<br>"
                    f"Round: {round_val}<br>"
                    f"GenType: {data.get('generation_type', 'N/A')}<br>"
                    f"Test: {data.get('test_score', 'N/A')}<br>"
                    f"Eval: {data.get('eval_score', 'N/A')}<br>"
                    f"Len: {data.get('token_length', 'N/A')}"
                )
                # Prepare the text to display on the node (e.g., id and test score).
                node_text[id] = f"Id: {data.get('id', 'N/A')}<br>Test: {data.get('test_score', 'N/A')}"
            
            # Add directed edges.
            G.add_edges_from(self.edges)
            
            # Determine unique rounds and assign a distinct color for each.
            unique_rounds = sorted({G.nodes[n].get("round_created", 0) for n in G.nodes()})
            num_rounds = len(unique_rounds)
            if num_rounds <= 20:
                colormap = plt.get_cmap("Pastel1")
            else:
                colormap = plt.get_cmap("Set3")
                    
            round_color_mapping = {
                r: "rgb({:.0f}, {:.0f}, {:.0f})".format(*(np.array(colormap(i/num_rounds)[:3]) * 255))
                for i, r in enumerate(unique_rounds)
            }
            node_colors = [round_color_mapping[G.nodes[n].get("round_created", 0)] for n in G.nodes()]
            
            # Generate layout.
            # Adjusting spring_layout parameters to improve spacing.
            pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)
            
            # Build node scatter trace.
            node_x, node_y = [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',  # Show markers and text.
                text=[node_text[node] for node in G.nodes()],
                textposition="middle center",
                marker=dict(
                    size=100,  # Adjust node size as needed.
                    color=node_colors,
                    line_width=2
                ),
                hoverinfo='text',
                hovertext=[node_hover_labels[node] for node in G.nodes()]
            )
            
            # Build arrow annotations for each edge.
            annotations = []
            offset_factor = 0.1  # Fraction of the edge length to shift.
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                dx = x1 - x0
                dy = y1 - y0
                d = np.sqrt(dx**2 + dy**2)
                if d > 0:
                    offset = offset_factor * d
                    new_x0 = x0 + (dx / d) * offset
                    new_y0 = y0 + (dy / d) * offset
                    new_x1 = x1 - (dx / d) * offset
                    new_y1 = y1 - (dy / d) * offset
                else:
                    new_x0, new_y0, new_x1, new_y1 = x0, y0, x1, y1
                
                annotations.append(dict(
                    ax=new_x0, ay=new_y0,
                    x=new_x1, y=new_y1,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#888",
                    standoff=2
                ))
            
            layout = go.Layout(
                title=dict(
                    text="Prompt Expansion Graph",
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=annotations,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            fig = go.Figure(data=[node_trace], layout=layout)
            
            # Save the interactive graph as an HTML file.
            html_filename = f"{filename}.html"
            fig.write_html(html_filename)
            self.logger.log(f"Interactive graph saved as {html_filename}")
    
        except Exception as e:
            self.logger.log(f"Error visualizing the graph: {e}")            