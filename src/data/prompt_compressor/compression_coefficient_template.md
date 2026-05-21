<|im_start|>system
# Task
Evaluate the input structured prompt by analyzing each node within its hierarchical structure and provide an importance rating for each non-header node.

## Step-by-Step Instructions:

1. **Read the Input Prompt Thoroughly**:
   - Begin by carefully reading the entire input prompt along with its specific details. Make sure to understand the task at hand, including any requirements or constraints provided.
   - `SystemPrompt` contains the instructions to solve the task in a hierarchical json format similar to tree structure, and `UserAssistantInteractions` provide some examples of how an assitant might provide answers for some task relevant queries provided by users, where assistant output follows output format usually specified in the `SystemPrompt`.

generic-lower or specific-higher 
2. **Identify top-level nodes**:
   Parse the entire input prompt tree and identify the top level nodes which do not have any parent nodes themselves.
    *  `top_level_nodes`: output the node ids in the list.

3. **Compression Analysis**:
   Iterate through each node in `top_level_nodes` and assign an importance rating. Consider all the children nodes in the sub tree while deciding the importance. Ensure the nodes are ranked by considering the importance of sibling nodes.
    * Explain the sub tree's importance (`explanation`): Start by summarizing the instructions in the sub tree and clearly state as per your understanding if:
      - it describes input or output format or any syntax.
      - it is explaining new concepts:
         - it has very detailed instructions which can be rephrased or summarized to be shorter. 
         - if the examples are redundant or too obvious and so dont need to be mentioned.
         - if any other modification can be done within ths subtree.
      - it is not adding any value and is redundant or generic:
         - if it has similar instructions compared to sibling `top_level_nodes`
         - if this sub tree can be merged or deleted.
    * Assign compression level (`compression_level`): 
      - Any input or output format must be marked as low compression.
      - Any sub tree that can be modified by reducing examples or summarizing, should be marked with medium compression.
      - Any sub tree that is is redundant so that it can be deleted or merged should be marked as high compression. 
      
## Input Format:

**Example Prompt**

```json
{"SystemPrompt": [{"NodeID":"<NodeID 1>", "Content": "<Heading 1>", "Children": [{"NodeID": "<NodeID 1.1>", "Content": "<Content 1.1>", "Children": [{"NodeID":"<NodeID 1.1.1>", "Content": ["<Example 1>", "<Example 2>", "<Example 3>"], "Children": []}]}, {"NodeID": "<NodeID 1.2>", "Content": "<Content 1.2>", "Children": [...]}]}, {"NodeID":"<NodeID 2>", "Content": "<Heading 2>", "Children": [...]}], "UserAssistantInteractions":[{"User": "<User 1>", "Assistant": "<Assistant 1>"}, {"User": ...}]}
```

## Output Format:

```json
{"top_level_nodes": ["NodeID 1", "NodeID 2"], "compression_analysis": [{"node_ID": "<NodeID 1>", "explanation": "<explanation for the compression level>", "compression_level": "<compression level of node in Low, Medium, High>"}, {"node_ID": "<NodeID 1.1>", "explanation": "<explanation for the rating>", "compression_level": "<compression level of node in Low, Medium, High>"} ...]}
```

<|im_end|>
<|im_start|>user
**Input Prompt**

```json
{parsed_prompt}
```

## Importance Rating Reminders:
- Correctly identify top-level nodes they dont have any parent nodes. Do not miss any top-level node.
- Ensure the nodes are relatively ranked: Determine the importance rating of top-level nodes based on sibling nodes. 
<|im_end|>
<|im_start|>assistant
```json