# LangGraph Structure Visualizer

A general-purpose tool that automatically detects and visualizes LangGraph structures from Python files or directories. Works with any LangGraph implementation.

## Features

🔍 **Auto-Detection**: Automatically finds Python files containing LangGraph code  
📊 **Structure Extraction**: Extracts nodes, edges, and conditional routing  
🎨 **Smart Visualization**: Creates professional graph diagrams with automatic layouts  
🌈 **Color Coding**: Different node types get distinct colors  
📁 **Batch Processing**: Can analyze entire directories at once  

## Installation

Requirements:
```bash
pip install matplotlib networkx
```

## Usage

### Basic Usage

```bash
# Analyze a single file
python langgraph_visualizer.py graph.py

# Analyze all LangGraph files in a directory
python langgraph_visualizer.py ./src/ --all

# Specify custom output file
python langgraph_visualizer.py graph.py --output my_graph.png
```

### Command Line Options

```bash
python langgraph_visualizer.py [path] [options]

Arguments:
  path                  Path to Python file or directory

Options:
  --file, -f           Specific Python file to analyze
  --directory, -d      Directory to search for LangGraph files
  --output, -o         Output PNG file path
  --all, -a            Visualize all found LangGraph files
  --help, -h           Show help message
```

## What It Detects

The tool automatically detects these LangGraph patterns:

### ✅ Supported Patterns

- **Graph Creation**: `StateGraph()` instances
- **Node Definition**: `add_node()` calls
- **Direct Edges**: `add_edge()` calls  
- **Conditional Routing**: `add_conditional_edges()` calls
- **START/END**: Entry and exit points

### 📋 Example Code Patterns

```python
# Graph builder detection
graph_builder = StateGraph(State)
builder = StateGraph(MyState)

# Node detection
graph_builder.add_node("classifier", classify_function)
graph_builder.add_node("agent", my_agent)

# Edge detection
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "agent")

# Conditional edge detection
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"option1": "agent1", "option2": "agent2"}
)
```

## Output

The tool generates:
- **PNG visualization** showing your graph structure
- **Console output** with detected components
- **Automatic layout** using NetworkX algorithms
- **Color-coded nodes** by type/function

## Color Coding

| Color | Node Type |
|-------|-----------|
| 🟢 Green | START/Entry points |
| 🔵 Blue | Classifiers |
| 🟠 Orange | Routers |
| 🟣 Purple | Agents/Data processors |
| 🔴 Red | Handlers/Processors |
| 🩷 Pink | NLP/Text processors |
| ⚫ Gray | Other/Generic |

## Example Output

After running on your project:

```
🚀 LangGraph Structure Visualizer
==================================================
📁 Found 1 LangGraph file(s)

🔍 Processing: ./tuned.py
🔍 Analyzing tuned.py...
📍 Found 7 nodes: ['START', 'classifier', 'router', 'data_agent', 'nl_sql_agent', 'continuation_handler', 'END']
🔗 Found 16 regular edges
🔀 Found 1 conditional edge sources
🎨 Creating nodes...
🔗 Creating edges...
🔀 Creating conditional edges...
✅ Visualization saved: tuned_langgraph_structure.png

🎉 LangGraph visualization complete!
```

## Use Cases

- 📖 **Documentation**: Generate diagrams for your LangGraph documentation
- 🐛 **Debugging**: Visualize complex graph flows to identify issues
- 👥 **Team Communication**: Share graph structures with team members
- 🏗️ **Architecture Review**: Analyze and review graph designs
- 📚 **Learning**: Understand existing LangGraph implementations

## Limitations

- Requires standard LangGraph patterns (may not detect heavily customized implementations)
- Works best with explicit `add_node`/`add_edge` calls
- Complex dynamic graph construction may not be fully captured
- Requires matplotlib and networkx for visualization

## Troubleshooting

**No graphs found**: Ensure your files contain LangGraph imports like:
```python
from langgraph.graph import StateGraph
```

**Layout issues**: The tool uses automatic layouts. For complex graphs, manual adjustment may be needed.

**Missing nodes**: Ensure you're using standard `add_node()` patterns.

## Contributing

This tool can be extended to support:
- More LangGraph patterns
- Different visualization styles
- Interactive HTML outputs
- Integration with other graph libraries

Created as a general-purpose tool for the LangGraph community! 🚀