import json
import uuid
import os
import re
import utils
import llm
import copy
import dirtyjson
import tiktoken
import random
def extract_prompt_segments(text):
    system_text, conversation, user_message = llm.segment_prompt(text)
    return {"System": system_text, "Conversation": conversation, "UserMessage": user_message}
    
class PromptNode:
    def __init__(self, node_id, parent_id, node_type, content, order):
        if node_id != 'System':
            if node_type.upper() not in ['H', 'P', 'L']:
                raise ValueError("Invalid node type. Must be one of: 'H', 'P', 'L'.")
        
        self.node_id = node_id
        self.parent_id = parent_id
        self.node_type = node_type
        self.content = content if isinstance(content, list) else str(content)
        self.order = order
        self.subnodes = []

class PromptTree:
    def __init__(self, id, text_prompt, model, provider, output_dir, parent_id=None, round_created=None, formatting=False):
        self.formatting = formatting
        self.existing_ids = set()
        self.model = model
        self.provider = provider
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback for models not in tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.output_dir = output_dir
        self.parent_id = parent_id
        self.id = id
        self.round_created = round_created
        self.token_length = None
        self.test_score = None
        self.eval_score = None
        self.eval_std = 0.0
        self.structure_template = "src/data/prompt_structure.md"
        self.format_template = "src/data/format_prompt.md"
        self.segment_template = "src/data/prompt_segment.md"

        extracted_segments = extract_prompt_segments(text_prompt)
        for key, value in extracted_segments.items():
            preview = str(value)[:100] if value else "[]"
            print(f"Prompt Tree {self.id} with {key}: {preview}...")

        structured_prompt_json = self.convert_to_prompt_json(extracted_segments['System'])
        
        self.tree = {
            "System": self.build_tree(structured_prompt_json),
            "Conversation": extracted_segments['Conversation'],
            "UserMessage": extracted_segments['UserMessage']
        }
    
    def get_textual_prompt(self):
        """ Returns the textual representation of the prompt tree. """
        return self.tree_to_text(self.tree["System"]), self.tree["Conversation"], self.tree["UserMessage"]

    def get_json_prompt(self):
        return {
            "System": self.tree_to_json(self.tree["System"]),
            "Conversation": self.tree["Conversation"],
            "UserMessage": self.tree["UserMessage"]
        }
    
    def get_prompt_chat_template(self):
        """ Returns the prompt in a chat template format. """
        system_prompt, conversation, user_message = self.get_textual_prompt()
        overall_prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        for turn in conversation:
            overall_prompt += f"<|im_start|>user\n{turn['User']}\n<|im_end|>\n<|im_start|>assistant\n{turn['Assistant']}\n<|im_end|>\n"
        overall_prompt += f"<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant"
        return overall_prompt
    
    def reset_scores(self):
        """Resets the evaluation scores and metadata for the prompt."""
        self.token_length = None
        self.test_score = None
        self.eval_score = None
        self.eval_std = 0.0
    
    def get_info_dict(self):
        """Returns a dictionary containing the prompt's attributes."""
        return {
            "id": self.id,
            "token_length": self.token_length,
            "test_score": self.test_score,
            "eval_score": self.eval_score,
            "eval_std": self.eval_std,
            "generation_type": "",
            "parent_id": self.parent_id,
            "round_created": self.round_created
        }

    def __deepcopy__(self, memo):
        """Custom deepcopy that shares the tiktoken encoding (stateless, 1-3 MB)
        and the lightweight per-tree id RNG instead of duplicating them.

        Without this, each `copy_and_reset()` copy carries its own tiktoken
        BPE table; with 4 candidates copied per round × 12 rounds that's
        ~150 MB of redundant encoders accumulating in the expansion_graph.
        Confirmed via tracemalloc: the standard deepcopy was the dominant
        per-round allocator (~20 MB / round in the smoke test, scaling with
        candidate count). See codebase_audit.md C3.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for key, value in self.__dict__.items():
            if key == "encoding":
                # tiktoken Encoding is read-only and thread-safe; sharing the
                # reference is correct.
                new_obj.encoding = value
            elif key == "_id_rng":
                # The per-tree RNG is meant to be unique per tree; the new
                # tree should get its own (lazy-initialised) RNG, so skip
                # copying the parent's RNG state.
                continue
            else:
                new_obj.__dict__[key] = copy.deepcopy(value, memo)
        return new_obj

    def copy_and_reset(self):
        """Creates a deep copy of the PromptTree while resetting scores and metadata.

        Uses our custom __deepcopy__ which shares the tiktoken encoding
        (stateless) instead of duplicating it — see above.
        """
        new_tree = copy.deepcopy(self)
        new_tree.reset_scores()
        return new_tree

    def create_candidate_copy(self):
        """
        Creates a fresh copy of a candidate while preserving all important metadata.
        This is the preferred method for creating copies in parallel processing scenarios.
        """
        candidate_copy = self.copy_and_reset()
        candidate_copy.id = self.id  # Preserve original ID for logging
        candidate_copy.parent_id = self.parent_id  # Preserve parent relationship
        candidate_copy.round_created = self.round_created  # Preserve round info
        candidate_copy.eval_score = self.eval_score  # Preserve evaluation scores
        candidate_copy.eval_std = self.eval_std  # Preserve evaluation standard deviation
        candidate_copy.token_length = self.token_length
        candidate_copy.test_score = self.test_score
        return candidate_copy       
    
    def calculate_token_length(self):
        """Calculates the token length of the system prompt."""
        system_prompt, conversation, user_message = self.get_textual_prompt()
        # For codex, the run_llm path spawns a CLI subprocess (~5-10s) just to
        # count tokens — wasteful. Use the local tiktoken encoder instead.
        if str(self.provider).lower() == "codex":
            text = (system_prompt or "")
            if conversation:
                for turn in conversation:
                    text += "\n" + str(turn.get("User", "")) + "\n" + str(turn.get("Assistant", ""))
            text += "\n" + (user_message or "")
            self.token_length = len(self.encoding.encode(text))
            return
        self.token_length = llm.run_llm(system_prompt, user_message, model=self.model, provider=self.provider, history=conversation, max_tokens=2, temperature=0)[2].prompt_tokens
    
    def generate_unique_id(self):
        # Use the tree's own seeded RNG (lazily initialised) so IDs are stable
        # across runs with the same seed regardless of cross-thread ordering.
        # Falling back to the global `random` module makes IDs depend on which
        # other call sites have advanced the global RNG state in this process.
        if not hasattr(self, "_id_rng"):
            try:
                seed_basis = hash(("prompt_id_rng", str(self.id))) & 0xFFFFFFFF
            except Exception:
                seed_basis = None
            self._id_rng = random.Random(seed_basis) if seed_basis is not None else random.Random()
        while True:
            new_id = "".join(self._id_rng.choices("0123456789ABCDEF", k=3))
            if new_id not in self.existing_ids:
                self.existing_ids.add(new_id)
                return new_id
    
    def build_tree(self, input_json):
        """
        Builds a tree from the given structured JSON.
        """
        root = PromptNode("System", None, "Root", "", 0)
        self._parse_json(input_json, root)  # Directly passing input_json as it's already a list
        return root

    def _parse_json(self, json_nodes, parent_node):
        """
        Recursively parses JSON and constructs a tree while preserving hierarchy.
        Avoids duplicate nodes and ensures correct subnode hierarchy.
        """
        seen_nodes = set()  # Track nodes to prevent duplicates

        for i, node in enumerate(json_nodes):
            # Extract the node ID (excluding "SubNodes")
            node_id = next((key for key in node.keys() if key != "SubNodes"), None)
            if not node_id:
                print(f"Skipping node {node} as it contains no valid key")
                continue

            value = node[node_id]

            try:
                node_parts = node_id.split('-')
                
                if len(node_parts) != 4:
                    raise ValueError(f"Invalid node key format: {node_id}")

                node_type, level, order, _ = node_parts
                new_id = f"{node_type}-{level}-{order}-{self.generate_unique_id()}"

                # Convert list values to string if necessary
                node_content = value if isinstance(value, list) else str(value)

                new_node = PromptNode(new_id, parent_node.node_id, node_type, node_content, int(order))
                parent_node.subnodes.append(new_node)

                # Recursively process subnodes if they exist
                if "SubNodes" in node and isinstance(node["SubNodes"], list) and node["SubNodes"]:
                    self._parse_json(node["SubNodes"], new_node)

            except ValueError as e:
                print(f"Skipping invalid key '{node_id}': {e}")

    def find_node(self, current_node, node_id):
        if current_node.node_id == node_id:
            return current_node
        for sub in current_node.subnodes:
            found = self.find_node(sub, node_id)
            if found:
                return found
        return None
    
    def find_parent_and_node(self, current_node, node_id):
        for sub in current_node.subnodes:
            if sub.node_id == node_id:
                return current_node, sub
            parent, found = self.find_parent_and_node(sub, node_id)
            if found:
                return parent, found
        return None, None          

    def build_node_to_parent_map(self, current_node=None, parent_map=None, parent_id=None):
        """
        Recursively builds a map from node_id to its parent_id for all nodes in the prompt tree.
        """
        if parent_map is None:
            parent_map = {}
        if current_node is None:
            current_node = self.tree["System"]

        for subnode in current_node.subnodes:
            parent_map[subnode.node_id] = current_node.node_id
            self.build_node_to_parent_map(subnode, parent_map, current_node.node_id)
        
        return parent_map

    def _update_node_orders(self, subnodes):
        """ Helper function to update orders and regenerate unique node IDs efficiently. """
        subnodes.sort(key=lambda x: x.order)
        for idx, node in enumerate(subnodes):
            node.order = idx
            node_type, level, _, guid = node.node_id.split('-')
            node.node_id = f"{node_type}-{level}-{idx}-{guid}"

    def collect_node_and_subnodes(self, node_ids):
        """ Collects the specified nodes and their immediate subnodes based on given node IDs. """
        result = set()
        
        # Process each specified node ID
        for node_id in node_ids:
            node = self.find_node(self.tree["System"], node_id)
            if node:
                # Add the node itself
                result.add(node.node_id)
                # Add only immediate subnodes, not recursive
                for subnode in node.subnodes:
                    result.add(subnode.node_id)
        
        return list(result)

    def update_all_node_orders(self):
        """ Updates the order of all subnodes in the tree. """
        def update_orders_recursive(node):
            self._update_node_orders(node.subnodes)
            for subnode in node.subnodes:
                update_orders_recursive(subnode)
        
        update_orders_recursive(self.tree["System"])

    def update_node_value(self, node_id, new_value):
        """ Updates the content of a given node by its node_id. """
        node = self.find_node(self.tree["System"], node_id)
        if not node:
            raise ValueError(f"Node Id {node_id} not found in the tree.")
        
        node.content = new_value if isinstance(new_value, list) else str(new_value)
    
    def reorder_node(self, node_id, new_order):
        """ Reorders a node within its parent's subnodes based on the new order. """
        parent_node, target_node = self.find_parent_and_node(self.tree["System"], node_id)
        if not target_node:
            raise ValueError(f"Node Id {node_id} not found in the tree.")
    
        # Remove target node temporarily
        parent_node.subnodes.remove(target_node)
    
        # Shift down nodes to maintain correct order
        for node in parent_node.subnodes:
            if node.order >= new_order:
                node.order += 1
    
        # Insert at new order position
        target_node.order = new_order
        parent_node.subnodes.append(target_node)
        
    def insert_node(self, parent_id, input_json):
        """
        Inserts a node into the prompt tree, ensuring all nested subnodes are correctly placed.
        """
        parent_node = self.find_node(self.tree["System"], parent_id)
        if not parent_node:
            raise ValueError(f"Parent node {parent_id} not found.")

        # Extract Node ID and its value (excluding "SubNodes")
        node_id = next((key for key in input_json.keys() if key != "SubNodes"), None)
        if not node_id:
            raise ValueError("Invalid input_json format: No valid node ID {node_id} found.")

        try:
            node_type, level, order, _ = node_id.split('-')
        except ValueError:
            node_type = node_id.split('-')[0]  # Fallback if format is not as expected
            level = node_id.split('-')[1]
            order = node_id.split('-')[2]

        if node_type.upper() not in ['H', 'P', 'L']:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of: 'H', 'P', 'L'.")

        if not level.isdigit() or not order.isdigit():
            raise ValueError(f"Invalid level or order in node ID: {node_id}. Both must be integers.")
            
        new_id = f"{node_type}-{level}-{order}-{self.generate_unique_id()}"
        node_content = input_json[node_id]

        # Create the new node and attach it to the parent
        new_node = PromptNode(new_id, parent_id, node_type, node_content if isinstance(node_content, list) else str(node_content), int(order))
        parent_node.subnodes.append(new_node)

        # If "SubNodes" exist, process them recursively
        if "SubNodes" in input_json and isinstance(input_json["SubNodes"], list):
            for subnode_json in input_json["SubNodes"]:
                self.insert_node(new_node.node_id, subnode_json)  # Recursive insertion
    
    def delete_node(self, node_id):
        """ Deletes a node and its subnodes from the prompt tree by node_id. """
        parent_node, node_to_delete = self.find_parent_and_node(self.tree["System"], node_id)
        if not node_to_delete or not parent_node:
            raise ValueError(f"Node Id {node_id} not found in the tree.")
    
        # Remove the node directly
        parent_node.subnodes.remove(node_to_delete)
        
    def merge_nodes(self, parent_id, node_id1, node_id2, merged_json):
        """ Merges two nodes under a parent node by deleting them and inserting a new merged node. """
        parent_node, node1 = self.find_parent_and_node(self.tree["System"], node_id1)
        _, node2 = self.find_parent_and_node(self.tree["System"], node_id2)
        
        if not parent_node or not node1 or not node2:
            raise ValueError(f"Invalid nodes {node_id1} or {node_id2} for merging.")
    
        # Delete nodes first
        self.delete_node(node_id1)
        self.delete_node(node_id2)
    
        # Insert merged node
        parent_id = parent_node.node_id
        self.insert_node(parent_id, merged_json)

    def update_subtree(self, node_id, new_subtree_json):
        """
        Replaces the subtree at a given node_id with a new subtree,
        preserving the root node_id.
        """
        node_to_update = self.find_node(self.tree["System"], node_id)
        if not node_to_update:
            raise ValueError(f"Node Id {node_id} not found in the tree.")

        # Extract content and subnodes from the new subtree json
        new_root_id = next((key for key in new_subtree_json if key != "SubNodes"), None)
        if not new_root_id:
            raise ValueError("Invalid new_subtree_json {new_root_id}: No valid root node found.")

        new_content = new_subtree_json[new_root_id]
        new_subnodes = new_subtree_json.get("SubNodes", [])

        # Update the content of the existing node
        self.update_node_value(node_id, new_content)

        # Clear existing subnodes
        node_to_update.subnodes.clear()

        # Insert new subnodes
        for subnode_json in new_subnodes:
            self.insert_node(node_id, subnode_json)

    def tree_to_json(self, node=None, is_root=True):
        """ Converts the tree structure into a JSON representation while avoiding empty subnodes and root content. """
        if node is None:
            node = self.tree["System"]

        json_output = {}

        # If it's not the root, store its content; otherwise, skip empty root content
        if not is_root or node.content.strip():
            json_output[node.node_id] = node.content

        # Only include subnodes if there are actual children
        if node.subnodes:
            json_output["SubNodes"] = [self.tree_to_json(sub, is_root=False) for sub in sorted(node.subnodes, key=lambda x: x.order)]

        return json_output if json_output else None  # Avoid returning empty dicts

    def tree_to_json_with_tokens(self, node=None, is_root=True):
        """ Converts the tree structure into a JSON representation with token counts for each node. """
        if node is None:
            node = self.tree["System"]

        json_output = {}

        # If it's not the root, store its content; otherwise, skip empty root content
        if not is_root or node.content.strip():
            json_output[node.node_id] = node.content
            
            content_str = str(node.content)
            num_tokens = len(self.encoding.encode(content_str))
            json_output["Tokens"] = num_tokens

        # Only include subnodes if there are actual children
        if node.subnodes:
            subnodes_json = [self.tree_to_json_with_tokens(sub, is_root=False) for sub in sorted(node.subnodes, key=lambda x: x.order)]
            subnodes_json = [s for s in subnodes_json if s]
            if subnodes_json:
                json_output["SubNodes"] = subnodes_json

        return json_output if json_output else None

    def tree_to_text(self, node=None, depth=0, is_first=True, parent_type=None):
        """ Converts the tree structure back to a well-formatted textual prompt while preserving existing prefixes. """
        if node is None:
            node = self.tree["System"]

        output = ""

        # Ensure correct indentation for list items
        indent = "    " * depth if parent_type == "L" else ""

        # Keep the original content without modifying prefixes
        content = node.content.rstrip()  # Remove only trailing spaces, keep internal `\n`

        # Handle headings correctly (use the existing `#` prefix if present)
        if node.node_type == "H":
            if not is_first and node.order > 0:
                output += "\n"  # Add extra line before new headings (except first heading)
            output += f"{content}\n"  # Preserve heading as is

        # Handle paragraphs correctly (without modifying their format)
        elif node.node_type == "P":
            output += f"{indent}{content}\n"  # Preserve all line breaks inside the paragraph

        # Handle lists correctly (preserve detected prefix without adding extra)
        elif node.node_type == "L":
            output += f"{indent}{content}\n"  # Content already has the correct prefix

        # Process subnodes in order
        for subnode in sorted(node.subnodes, key=lambda x: x.order):
            output += self.tree_to_text(subnode, depth + 1, is_first=False, parent_type=node.node_type)

        return output

    def redact_tree_for_nodes(self, node_ids):
        """
        Returns a redacted version of the prompt tree where:
        - Target nodes and their subnodes remain fully visible.
        - Headings (H) always retain content.
        - All other nodes have their content replaced with '<redacted>'.
        - No nodes are removed; only their content is altered.
        """
        redacted_tree = copy.deepcopy(self)
        system_node = redacted_tree.tree["System"]

        def dfs_redact(node, is_target_ancestor=False):
            is_target = node.node_id in node_ids
            is_heading = node.node_type.upper() == "H"
            
            # Propagate target ancestry
            is_target_ancestor = is_target or is_target_ancestor  

            # Process subnodes first (DFS traversal)
            for subnode in node.subnodes:
                dfs_redact(subnode, is_target_ancestor)

            # Redact non-target, non-heading nodes that are not ancestors of a target
            if not (is_target or is_heading or is_target_ancestor):
                node.content = "<redacted>"

        # Start DFS redaction from the system node
        dfs_redact(system_node)

        return redacted_tree

    def count_words(self, text):
        """ Counts the number of words in a given text. """
        return len(text.split())
    
    def parse_prompt_into_nodes(self, prompt_text):
        """ Parses the prompt into structured nodes while preserving hierarchy. """
        nodes = []
        current_node = None
        lines = prompt_text.split("\n")

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("#"):  # Heading Detected
                if current_node:
                    nodes.append(current_node)
                current_node = {"type": "H", "content": stripped_line, "subnodes": []}

            elif stripped_line.startswith("-") or stripped_line.startswith("*"):  # List Item Detected
                if current_node and isinstance(current_node, dict) and current_node.get("type") in ["H", "L"]:
                    current_node.setdefault("subnodes", []).append({"type": "L", "content": stripped_line})
                else:
                    if current_node:
                        nodes.append(current_node)
                    current_node = {"type": "L", "content": stripped_line, "subnodes": []}

            else:  # Paragraph or Description (Including Examples)
                if current_node and isinstance(current_node, dict):
                    current_node.setdefault("subnodes", []).append({"type": "P", "content": stripped_line})
                else:
                    current_node = {"type": "P", "content": stripped_line, "subnodes": []}

        if current_node:
            nodes.append(current_node)

        return nodes
    
    def smart_chunk_prompt(self, prompt_text, max_chunk_size=4096):
        """ Smartly chunks the input prompt while preserving structure. """
        parsed_nodes = self.parse_prompt_into_nodes(prompt_text)
        chunks = []
        current_chunk = []
        current_size = 0

        for node in parsed_nodes:
            node_size = len(node["content"])  # Approximate size of the node

            if current_size + node_size > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(node)
            current_size += node_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    def format_prompt(self, init_prompt):
        template = open(self.format_template, "r", encoding="utf-8").read()
        prompt = template.replace("#InitialPrompt#", init_prompt)
        with open(os.path.join(self.output_dir, str(self.id) + "_prompt_formatting_request.txt"), "w") as f:
            f.write(prompt)
        _, formatted_prompt, _ = llm.infer(prompt,  max_tokens=5000, temperature=0, top_p=1.0, model=self.model, provider=self.provider)
        formatted_prompt = utils.clean_llm_response(formatted_prompt)
        with open(os.path.join(self.output_dir, str(self.id) + "_formatted_prompt.txt"), "w") as f:
            f.write(formatted_prompt)
        return formatted_prompt

    def convert_to_prompt_json(self, init_prompt, char_threshold=1000):
        """
        Recursively structure a Markdown prompt into an H/P/L tree,
        splitting by parse_markdown_sections() and only using the LLM
        to structure leaf chunks (<= char_threshold), including the
        heading line in each leaf so the heading node itself is emitted.
        """
        # Preload the structuring template once
        struct_tpl = open(self.structure_template, "r", encoding="utf-8").read()

        def _struct_chunk(content, node_id):
            prompt = struct_tpl.replace("{#parsed_prompt#}", content)
            req_file = f"{self.id}_{node_id}_structuring_request.txt"
            res_file = f"{self.id}_{node_id}_structuring_response.txt"
            raw_res_file = f"{self.id}_{node_id}_structuring_raw_response.txt"

            # Save request
            with open(os.path.join(self.output_dir, req_file), "w", encoding="utf-8") as f:
                f.write(prompt)

            # Call LLM
            _, raw, _ = llm.infer(
                prompt,
                max_tokens=5000,
                temperature=0,
                top_p=0.01,
                model=self.model,
                provider=self.provider
            )
            with open(os.path.join(self.output_dir, raw_res_file), "w", encoding="utf-8") as f:
                f.write(raw)
            
            cleaned = utils.clean_llm_response(raw)

            # Save response
            with open(os.path.join(self.output_dir, res_file), "w", encoding="utf-8") as f:
                f.write(cleaned)

            try:
                parsed = dirtyjson.loads(cleaned)
                return parsed if isinstance(parsed, list) else [parsed]
            except ValueError as e:
                print(f"❌ JSON parse error structuring '{node_id}': {e}")
                return []

        def _process_chunk(text, node_id, heading_text=None, level=1):
            # Debug
            print(f"Node {node_id} ─ chars: {len(text)}")

            # Leaf chunk: prefix with its heading if given, or if no headings present
            if len(text) <= char_threshold or "#" not in text:
                content = text if heading_text is None else f"{heading_text}\n{text}"
                return _struct_chunk(content, node_id)

            # Otherwise, segment locally using the parser
            segments = utils.parse_markdown_sections(text)

            # Save raw segments for debugging
            seg_file = f"{self.id}_{node_id}_segments.txt"
            with open(os.path.join(self.output_dir, seg_file), "w", encoding="utf-8") as f:
                f.write(json.dumps(segments, indent=2, ensure_ascii=False))

            nodes = []
            for seg in segments:
                hdr_id = next(k for k in seg if k.startswith("H-"))
                hdr_text = seg[hdr_id]
                subtext = seg.get("SubText", "")

                # Recurse on subtext, passing heading to structuring
                subtree = _process_chunk(subtext, hdr_id, heading_text=hdr_text, level=level+1)

                # Always extend with the returned subtree
                nodes.extend(subtree)

            return nodes

        # Prepare the full prompt text
        full_text = self.format_prompt(init_prompt) if self.formatting else init_prompt
        tree = _process_chunk(full_text, "full_prompt", heading_text=None, level=1)

        # Serialize and save final structured tree
        tree_str = json.dumps(tree, indent=4, ensure_ascii=False)
        out_file = os.path.join(self.output_dir, f"{self.id}_structured_prompt.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(tree_str)

        return tree

