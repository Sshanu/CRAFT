<|im_start|>system
# Task  
Restructure the descriptive prompt into a JSON structure using unique Node IDs for each element.

## Instruction About the Task  
Your goal is to convert a descriptive prompt into a structured JSON format. Each content element—headings, paragraphs, list items be represented as a node using the format:  
`<NodeType>-<Level>-<Order>-<GUID>`

- **NodeType:**  
  - **H:** Heading  
  - **P:** Paragraph  
  - **L:** List item  
- **Level:** Represents the depth in the hierarchy:  
  - *Level 1:* Top-level nodes (e.g., main headings).  
  - *Level 2:* Direct children of Level 1 nodes (e.g., paragraphs or list items under a heading).  
  - *Level 3 and beyond:* Subsequent nested nodes.
- **Order:** The sequential position of the node among its siblings at the same level (starting from 0).
- **GUID:** A randomized, exactly 3-character string (using characters 0-9 and A-Z) that uniquely identifies the node. **No two nodes can share the same GUID.**

**Important Notes:**  
- **Heading Variations:**  
  - Headings may be explicitly denoted using one or more `#` symbols (e.g., `#` for a main heading, `##` for a subheading, `####` for a deeper subheading).  
  - Headings can also be implied by context without any `#` symbols. In both cases, treat them as heading nodes (NodeType **H**) and assign the appropriate level.

- **Distinguishing Headings from Lists:**  
  - **Headings:** Are standalone title sections. They may include numbers (e.g., `# 1. Text` or `1. Text`) and are usually followed by a paragraph describing the section.  
  - **Lists:** Are enumerations. For example, ordered lists like:  
    ```
    1. paragraph  
    2. paragraph
    ```  
    or unordered lists like:  
    ```
    * paragraph  
    * paragraph
    ```  
    Do not confuse headings with list items. Headings should be captured as NodeType **H** and list items as NodeType **L**.

- Use **SubNodes** as the key to nest nodes within their parent.

## Example Prompt
The prompt is enclosed within <prompt_start> and <prompt_end> tags.

<prompt_start>
Introduction  
Welcome to our documentation.  

# Features  
- Feature One  
  Description: This is the first feature.  
  Examples: "Feature A"  

## Advanced Features  
- Feature Two  
  Description: This is the second feature.  
  Examples: "Feature B", "Feature C"  
  * Sub Feature: Detailed view available.  

# Code Block Example  
Here is a simple Python function:  
```
def greet(name):  
    return f"Hello, {name}!"  
```
Use this function to generate personalized greetings.  
<prompt_end>

*Note:*  
- The first heading, "Introduction", is implicitly a heading even though it lacks a `#` symbol.  
- The code block should be preserved properly within the structured output.

---

## Expected Output  
```json
[
    {
        "H-1-0-7K9": "Introduction",
        "SubNodes": [
            {
                "P-2-0-B8Z": "Welcome to our documentation."
            }
        ]
    },
    {
        "H-1-1-3F2": "# Features",
        "SubNodes": [
            {
                "L-2-0-M5T": "- Feature One",
                "SubNodes": [
                    {
                        "P-3-0-Q1V": "Description: This is the first feature."
                    },
                    {
                        "P-3-1-R8C": "Examples: \"Feature A\""
                    }
                ]
            },
            {
                "H-2-0-P4X": "## Advanced Features",
                "SubNodes": [
                    {
                        "L-2-1-D6Y": "- Feature Two",
                        "SubNodes": [
                            {
                                "P-3-0-Z3L": "Description: This is the second feature."
                            },
                            {
                                "P-3-1-U7J": "Examples: \"Feature B\", \"Feature C\""
                            },
                            {
                                "L-4-2-F1Q": "* Sub Feature: Detailed view available."
                            }
                        ]
                    }
                ]
            }
        ]
    },
    {
        "H-1-2-X1Y": "# Code Block Example",
        "SubNodes": [
            {
                "P-2-0-T7M": "Here is a simple Python function:\n```\ndef greet(name):\n    return f\"Hello, {name}!\"\n```\nUse this function to generate personalized greetings."
            }
        ]
    }
]
```

*Note:* In the expected output, each GUID (e.g., "7K9", "B8Z", etc.) is a random, unique 3-character string composed of characters from 0-9 and A-Z.
<|im_end|>
<|im_start|>user
**Input Prompt**

<prompt_start>
{#parsed_prompt#}
<prompt_end>


## Final Guidelines  
- **Preserve Original Text:** Do not modify original headings, paragraphs, or list items.

- **Node ID Format:** Use `<NodeType>-<Level>-<Order>-<GUID>` exactly.

- **Unique GUIDs:** Assign unique 3-character GUIDs (0-9, A-Z) to each node.

- **Maintain Hierarchy:** Nest nodes correctly using **SubNodes**. Correctly identify which nodes are **SubNodes**, do not simply add at the same level.

- **Capture All Headings:**  
  - Explicit headings (`#`, `##`, etc.) as NodeType **H** with correct levels.  
  - Implicit headings (e.g., `Introduction`) as NodeType **H**, Level 1 if top-level.

- **Differentiate Headings from Lists:**  
  - **Headings:** Titles possibly numbered, followed by paragraphs.  
  - **Lists:** Enumerations indicated by numbers or bullets (`1.`, `*`, etc.).

- **Paragraphs and Code Blocks:**  
  - Combine related multi-line paragraphs into single nodes, preserving newlines.  
  - Capture multi-line code blocks (denoted by ```<code>```) as single paragraph nodes (NodeType **P**), preserving formatting and newlines.

- **Validate Structure:** Ensure JSON strictly matches provided schema.
<|im_end|>
<|im_start|>assistant
```json