#!/usr/bin/env python3
"""
LangGraph Structure Visualizer

A general-purpose tool that automatically detects and visualizes LangGraph structures
from Python files or directories. Works with any LangGraph implementation.

Usage:
    python langgraph_visualizer.py /path/to/file.py
    python langgraph_visualizer.py /path/to/directory/
    python langgraph_visualizer.py --file graph.py --output my_graph.png
"""

import argparse
import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import networkx as nx
from dataclasses import dataclass

@dataclass
class GraphStructure:
    """Data class to hold extracted graph structure"""
    nodes: List[str]
    edges: List[Tuple[str, str, Dict]]
    conditional_edges: Dict[str, List[Tuple[str, Dict]]]
    imports: Set[str]
    graph_builders: List[str]
    file_path: str

class LangGraphDetector:
    """Detects LangGraph structures in Python files"""
    
    def __init__(self):
        self.langgraph_patterns = {
            'imports': [
                r'from\s+langgraph',
                r'import\s+langgraph',
                r'from\s+langgraph\.graph\s+import\s+StateGraph',
                r'StateGraph'
            ],
            'graph_builder': [
                r'(\w+)\s*=\s*StateGraph\(',
                r'graph_builder\s*=\s*StateGraph\(',
                r'builder\s*=\s*StateGraph\('
            ],
            'add_node': [
                r'(\w+)\.add_node\(\s*["\'](\w+)["\']',
                r'\.add_node\(\s*["\'](\w+)["\']'
            ],
            'add_edge': [
                r'\.add_edge\(\s*(START|END|["\'][\w_]+["\'])\s*,\s*["\'](\w+)["\']',
                r'\.add_edge\(\s*(["\'][\w_]+["\'])\s*,\s*(START|END|["\'][\w_]+["\'])',
                r'add_edge\(\s*(START|END)\s*,\s*["\'](\w+)["\']',
                r'add_edge\(\s*["\'](\w+)["\'].*?["\'](\w+)["\']'
            ],
            'conditional_edges': [
                r'(\w+)\.add_conditional_edges\(\s*["\'](\w+)["\'].*?\{([^}]+)\}',
                r'\.add_conditional_edges\(\s*["\'](\w+)["\'].*?\{([^}]+)\}'
            ]
        }
    
    def find_langgraph_files(self, path: str) -> List[str]:
        """Find Python files that contain LangGraph imports"""
        langgraph_files = []
        
        if os.path.isfile(path):
            if self.contains_langgraph(path):
                langgraph_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if self.contains_langgraph(file_path):
                            langgraph_files.append(file_path)
        
        return langgraph_files
    
    def contains_langgraph(self, file_path: str) -> bool:
        """Check if file contains LangGraph imports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in self.langgraph_patterns['imports']:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            return False
        except Exception:
            return False
    
    def extract_graph_structure(self, file_path: str) -> Optional[GraphStructure]:
        """Extract LangGraph structure from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"🔍 Analyzing {os.path.basename(file_path)}...")
            
            # Extract imports
            imports = self._extract_imports(content)
            
            # Extract graph builders
            graph_builders = self._extract_graph_builders(content)
            
            # Extract nodes
            nodes = self._extract_nodes(content)
            
            # Extract edges
            edges = self._extract_edges(content)
            
            # Extract conditional edges
            conditional_edges = self._extract_conditional_edges(content)
            
            # Add START and END if not present but referenced
            if any('START' in str(edge) for edge in edges) and 'START' not in nodes:
                nodes.insert(0, 'START')
            if any('END' in str(edge) for edge in edges) and 'END' not in nodes:
                nodes.append('END')
            
            print(f"📍 Found {len(nodes)} nodes: {nodes}")
            print(f"🔗 Found {len(edges)} regular edges")
            print(f"🔀 Found {len(conditional_edges)} conditional edge sources")
            
            return GraphStructure(
                nodes=nodes,
                edges=edges,
                conditional_edges=conditional_edges,
                imports=imports,
                graph_builders=graph_builders,
                file_path=file_path
            )
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return None
    
    def _extract_imports(self, content: str) -> Set[str]:
        """Extract LangGraph-related imports"""
        imports = set()
        for pattern in self.langgraph_patterns['imports']:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.update(matches if isinstance(matches, list) else [matches])
        return imports
    
    def _extract_graph_builders(self, content: str) -> List[str]:
        """Extract graph builder variable names"""
        builders = []
        for pattern in self.langgraph_patterns['graph_builder']:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                if isinstance(matches[0], str):
                    builders.extend(matches)
                else:
                    builders.extend([m for m in matches if isinstance(m, str)])
        return list(set(builders))  # Remove duplicates
    
    def _extract_nodes(self, content: str) -> List[str]:
        """Extract node names from add_node calls"""
        nodes = []
        for pattern in self.langgraph_patterns['add_node']:
            matches = re.findall(pattern, content)
            if matches:
                # Handle different match groups
                if isinstance(matches[0], tuple):
                    nodes.extend([match[1] if len(match) > 1 else match[0] for match in matches])
                else:
                    nodes.extend(matches)
        return list(set(nodes))  # Remove duplicates
    
    def _extract_edges(self, content: str) -> List[Tuple[str, str, Dict]]:
        """Extract edges from add_edge calls"""
        edges = []
        for pattern in self.langgraph_patterns['add_edge']:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    # Get the last two elements as source and target
                    if len(match) == 2:
                        source, target = match
                    else:
                        source, target = match[-2], match[-1]
                    
                    # Clean up source and target
                    source = source.strip('"\' ')
                    target = target.strip('"\' ')
                    edges.append((source, target, {}))
                elif isinstance(match, str):
                    # Single match case - try to parse it
                    pass
        return edges
    
    def _extract_conditional_edges(self, content: str) -> Dict[str, List[Tuple[str, Dict]]]:
        """Extract conditional edges"""
        conditional_edges = {}
        
        for pattern in self.langgraph_patterns['conditional_edges']:
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    source = match[-2] if len(match) >= 2 else match[0]
                    conditions_str = match[-1]
                    
                    # Parse the conditions dictionary
                    condition_pattern = r'["\'](\w+)["\']:\s*["\'](\w+)["\']'
                    condition_matches = re.findall(condition_pattern, conditions_str)
                    
                    conditional_edges[source] = []
                    for condition, target in condition_matches:
                        conditional_edges[source].append((target, {'condition': condition}))
        
        return conditional_edges

class LangGraphVisualizer:
    """Creates visualizations from LangGraph structures"""
    
    def __init__(self):
        self.default_colors = {
            'START': '#4CAF50',
            '__start__': '#4CAF50',
            'classifier': '#2196F3',
            'router': '#FF9800',
            'agent': '#9C27B0',
            'handler': '#F44336',
            'END': '#607D8B',
            '__end__': '#607D8B',
            'default': '#757575'
        }
    
    def create_visualization(self, graph_structure: GraphStructure, output_path: str = None) -> str:
        """Create a visualization from the graph structure"""
        
        if not graph_structure or not graph_structure.nodes:
            print("❌ No graph structure to visualize")
            return None
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Calculate positions
        positions = self._calculate_positions(graph_structure.nodes, graph_structure.edges)
        
        # Create nodes
        print("🎨 Creating nodes...")
        for node in graph_structure.nodes:
            color = self._get_node_color(node)
            self._create_node(ax, positions[node], node, color)
        
        # Create regular edges
        print("🔗 Creating edges...")
        for source, target, edge_data in graph_structure.edges:
            if source in positions and target in positions:
                self._create_arrow(ax, positions[source], positions[target])
        
        # Create conditional edges
        print("🔀 Creating conditional edges...")
        for source, targets in graph_structure.conditional_edges.items():
            if source in positions:
                for target, edge_data in targets:
                    if target in positions:
                        condition = edge_data.get('condition', '')
                        self._create_arrow(ax, positions[source], positions[target], 
                                         condition, color='blue')
        
        # Add title and metadata
        file_name = os.path.basename(graph_structure.file_path)
        plt.title(f'LangGraph Structure - {file_name}', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        self._add_legend(ax, graph_structure.nodes)
        
        # Add info box
        self._add_info_box(ax, graph_structure)
        
        # Save the visualization
        if not output_path:
            base_name = os.path.splitext(os.path.basename(graph_structure.file_path))[0]
            output_path = f"{base_name}_langgraph_structure.png"
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        return output_path
    
    def _calculate_positions(self, nodes: List[str], edges: List[Tuple]) -> Dict[str, Tuple[float, float]]:
        """Calculate optimal positions for nodes"""
        try:
            # Create networkx graph for layout
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            
            # Add edges for layout calculation
            for source, target, _ in edges:
                if source in nodes and target in nodes:
                    G.add_edge(source, target)
            
            # Try hierarchical layout first
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Fallback to spring layout
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            
            # Normalize positions
            positions = {}
            for node, (x, y) in pos.items():
                positions[node] = (
                    (x - min(pos.values(), key=lambda p: p[0])[0]) / 
                    (max(pos.values(), key=lambda p: p[0])[0] - min(pos.values(), key=lambda p: p[0])[0] or 1) * 10 + 2,
                    (y - min(pos.values(), key=lambda p: p[1])[1]) / 
                    (max(pos.values(), key=lambda p: p[1])[1] - min(pos.values(), key=lambda p: p[1])[1] or 1) * 6 + 2
                )
            
            return positions
            
        except Exception as e:
            print(f"⚠️ Layout calculation failed: {e}, using simple layout")
            # Simple fallback layout
            positions = {}
            cols = 3
            for i, node in enumerate(nodes):
                positions[node] = (
                    2 + (i % cols) * 4,
                    8 - (i // cols) * 2
                )
            return positions
    
    def _get_node_color(self, node: str) -> str:
        """Get color for a node based on its name/type"""
        node_lower = node.lower()
        
        for key, color in self.default_colors.items():
            if key.lower() in node_lower:
                return color
        
        # Special cases
        if any(word in node_lower for word in ['sql', 'data', 'query']):
            return '#9C27B0'  # Purple for data/sql nodes
        if any(word in node_lower for word in ['nlp', 'language', 'text']):
            return '#E91E63'  # Pink for NLP nodes
        if any(word in node_lower for word in ['continue', 'handler', 'process']):
            return '#F44336'  # Red for handlers
        
        return self.default_colors['default']
    
    def _create_node(self, ax, pos: Tuple[float, float], text: str, color: str, width: float = 2.0, height: float = 0.8):
        """Create a visual node"""
        x, y = pos
        
        # Create rounded rectangle
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                           boxstyle="round,pad=0.1", 
                           facecolor=color, 
                           edgecolor='black',
                           linewidth=2)
        ax.add_patch(box)
        
        # Add text with line wrapping
        display_text = text.replace('_', '\n') if len(text) > 10 and '_' in text else text
        ax.text(x, y, display_text, ha='center', va='center', 
               fontsize=9, fontweight='bold', 
               color='white' if color != '#FFEB3B' else 'black',
               wrap=True)
    
    def _create_arrow(self, ax, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                     text: str = '', color: str = 'black'):
        """Create an arrow between nodes"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Create arrow
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle="->", shrinkA=50, shrinkB=50,
                              mutation_scale=20, fc=color, ec=color,
                              linewidth=2)
        ax.add_artist(arrow)
        
        # Add label if provided
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, text, ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor='white', edgecolor='gray', alpha=0.9))
    
    def _add_legend(self, ax, nodes: List[str]):
        """Add color legend"""
        unique_colors = set(self._get_node_color(node) for node in nodes)
        legend_elements = []
        
        color_labels = {
            '#4CAF50': 'Start/Entry',
            '#2196F3': 'Classifier', 
            '#FF9800': 'Router',
            '#9C27B0': 'Agents/Data',
            '#F44336': 'Handlers',
            '#E91E63': 'NLP/Text',
            '#607D8B': 'End/Exit',
            '#757575': 'Other'
        }
        
        for color in unique_colors:
            if color in color_labels:
                legend_elements.append(patches.Patch(color=color, label=color_labels[color]))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def _add_info_box(self, ax, graph_structure: GraphStructure):
        """Add information box with graph details"""
        info_text = f"""Graph Analysis:
• File: {os.path.basename(graph_structure.file_path)}
• Nodes: {len(graph_structure.nodes)}
• Regular Edges: {len(graph_structure.edges)}
• Conditional Edges: {len(graph_structure.conditional_edges)}
• Graph Builders: {', '.join(graph_structure.graph_builders) if graph_structure.graph_builders else 'Detected'}
"""
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor='lightblue', alpha=0.8))

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Automatically detect and visualize LangGraph structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python langgraph_visualizer.py graph.py
    python langgraph_visualizer.py ./src/
    python langgraph_visualizer.py --file graph.py --output my_visualization.png
    python langgraph_visualizer.py --directory ./project/ --all
        """
    )
    
    parser.add_argument('path', nargs='?', help='Path to Python file or directory')
    parser.add_argument('--file', '-f', help='Specific Python file to analyze')
    parser.add_argument('--directory', '-d', help='Directory to search for LangGraph files')
    parser.add_argument('--output', '-o', help='Output PNG file path')
    parser.add_argument('--all', '-a', action='store_true', help='Visualize all found LangGraph files')
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.path or args.file or args.directory
    if not input_path:
        input_path = input("Enter path to Python file or directory: ").strip()
    
    if not input_path or not os.path.exists(input_path):
        print("❌ Invalid or non-existent path")
        return
    
    print("🚀 LangGraph Structure Visualizer")
    print("=" * 50)
    
    # Initialize detector and visualizer
    detector = LangGraphDetector()
    visualizer = LangGraphVisualizer()
    
    # Find LangGraph files
    langgraph_files = detector.find_langgraph_files(input_path)
    
    if not langgraph_files:
        print("❌ No LangGraph structures found in the specified path")
        return
    
    print(f"📁 Found {len(langgraph_files)} LangGraph file(s)")
    
    # Process files
    for file_path in langgraph_files:
        print(f"\n🔍 Processing: {file_path}")
        
        # Extract structure
        structure = detector.extract_graph_structure(file_path)
        if not structure:
            continue
        
        # Create visualization
        output_path = args.output if len(langgraph_files) == 1 else None
        result_path = visualizer.create_visualization(structure, output_path)
        
        if result_path:
            print(f"✅ Visualization saved: {result_path}")
        
        if not args.all and len(langgraph_files) > 1:
            # Ask if user wants to continue with next file
            response = input(f"\nContinue with next file? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                break
    
    print("\n🎉 LangGraph visualization complete!")

if __name__ == "__main__":
    main()