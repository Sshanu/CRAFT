<|im_start|>system
### **Instruction for Extracting Answer and Explanation**
1. Identify if the response is in **JSON format**. If so:
   - Extract the key that holds the **answer** (e.g., `"answer"`, `"result"`, `"output"`, or similar).
   - Extract the key that holds the **explanation** (if any) and separate it.
   - If the explanation is found in the JSON, ensure it's removed from the **answer** field.

2. If the response is in **list format**, check:
   - If the first item appears to be the **answer**, and the second item appears to be an **explanation**.
   - If only one item exists, assume it's the **answer**, and set **explanation** as `""`.

3. If the response is a **direct string** or separated by **symbols (e.g., `---`, `||`, `:`)**:
   - If a separator (like `---`, `||`, `:`) is present, assume the **first part** is the **answer** and the **second part** is the **explanation**.
   - If no clear separation exists, assume the entire response is the **answer** and set **explanation** to `""`.

4. Output the final result in JSON format:
   ```json
   {"answer": <extracted answer>, "explanation": <extracted explanation>}
   ```

---

### **Examples**

**Input:**
```json
{
  "answer": "{"key1":"value1", "key2":"value2"}",
  "explanation": "This is the meaning of life."
}
```
**Output:**
```json
{"answer": "{"key1":"value1", "key2":"value2"}", "explanation": "This is the meaning of life."}
```

**Input:**
```json
{
  "result": "Yes|India",
  "details": "Because the conditions are met."
}
```
**Output:**
```json
{"answer": "Yes|India", "explanation": "Because the conditions are met."}
```

**Input:**
```
["Paris", "It is the capital of France."]
```
**Output:**
```
{"answer": "Paris", "explanation": "It is the capital of France."}
```

**Input:**
```
["Elephant", "Zebra"]
```
**Output:**
```json
{"answer": '["Elephant", "Zebra"]', "explanation": ""}
```

**Input:**
```
The answer is 7.
```
**Output:**
```
{"answer": "The answer is 7.", "explanation": ""}
```

**Input:**
```
5 --- It is a prime number.
```
**Output:**
```
{"answer": "5", "explanation": "It is a prime number."}
```

**Input:**
```
No: This is incorrect due to missing evidence.
```
**Output:**
```json
{"answer": "No", "explanation": "This is incorrect due to missing evidence."}
```

**Input:**
```
<Label>Positive</Label> <Reasoning>Reasoniong for answer ...</Reasoning>
```
**Output:**
```json
{"answer": "Positive", "explanation": "Reasoniong for answer ..."}
```

**Input:**
```
Label: Negative - Reasoning: reasoning for negative ...
```
**Output:**
```json
{"answer": "Negative", "explanation": "reasoning for negative ..."}
```
<|im_end|>
<|im_start|>user
**Input:**
{Response}
<|im_end|>
<|im_start|>assistant
```json