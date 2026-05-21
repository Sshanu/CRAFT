<|im_start|>system
Given the Markdown document enclosed between `<prompt_start>` and `<prompt_end>`, split it into sections at every Markdown heading of the **shallowest depth** (i.e. the fewest `#` characters) present in the document. A section begins at each such heading line; any text before the first shallow‐depth heading or after the last is also a section.

Produce a JSON array where each element is an object with exactly two keys:

1. **Node ID** – the key name itself, using the format `H-1-<Order>-<GUID>`:

   * `H`: NodeType (always headings in this output)
   * `1`: Level (treat each section as a top-level node)
   * `<Order>`: zero-based index of the section in document order
   * `<GUID>`: unique 3-character string (`0–9` or `A–Z`)

   The Node ID’s **value** must be the exact heading line (including its `#` characters), or `"None"` if the section has no heading at that shallow depth.

2. **SubText** – a string of all the Markdown content under that section (excluding the heading line), up to the next shallow‐depth heading, trimmed of leading/trailing whitespace.

## How it works

* First scan all headings (`#`, `##`, `###`, etc.) to determine the **minimum heading level** (fewest `#`) present.
* Split only at that level. Deeper headings become part of the `SubText`.
* If there are level-1 (`#`) headings, use those. If none, fall back to level-2 (`##`), and so on.

### Example 1

Has both `#` and `##` headings; minimum level is `#`, so split at `#` only:

```json
[
  {
    "H-1-0-A7F": "# Introduction",
    "SubText": "Welcome to our documentation"
  },
  {
    "H-1-1-B3X": "# Features",
    "SubText": "- Feature One\n\t…\n\n## Advanced Features\n\t- Feature Two…"
  },
  {
    "H-1-2-Q9Z": "# Code Block Example",
    "SubText": "Here is a simple Python function:…"
  }
]
```

### Example 2

No `#` headings; minimum level is `##`, so split at `##`:

```json
[
  {
    "H-1-0-B6U": "None",
    "SubText": "- Feature One…"
  },
  {
    "H-1-1-3RT": "## Advanced Features",
    "SubText": "- Feature Two…"
  },
  {
    "H-1-2-ZQ1": "None",
    "SubText": "There are some other features as well."
  }
]
```
<|im_end|>
<|im_start|>user
**Input Prompt**

<prompt_start>
{#parsed_prompt#}
<prompt_end>

Reminder: Identify the minimum heading level in the entire document (the least number of #) and split only at that level. Treat all deeper headings as part of the section’s content.
Note: You must ensure no text or information is lost.
You must not 
<|im_end|>
<|im_start|>assistant
```json