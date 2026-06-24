
    
import javalang
import pandas as pd
import os
import sys

def extract_java_features(java_path):
    """
    Parses a Java file to extract structural AST nodes and semantic code tokens.
    """
    try:
        with open(java_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # 1. AST Extraction
        tree = javalang.parse.parse(code)
        ast_nodes = []
        for _, node in tree:
            if isinstance(node, (javalang.tree.Declaration, 
                                 javalang.tree.MethodInvocation, 
                                 javalang.tree.Statement)):
                node_type = type(node).__name__
                node_name = getattr(node, 'name', None)
                
                # Only append name if it exists, otherwise just keep the node type
                if node_name:
                    ast_nodes.append(f"{node_type}_{node_name}")
                else:
                    ast_nodes.append(node_type)
        
        # 2. Token Extraction
        tokens = [t.value for t in javalang.tokenizer.tokenize(code) 
                  if not isinstance(t, (javalang.tokenizer.Separator, javalang.tokenizer.Operator))]
        
        if ast_nodes and tokens:
            return " ".join(ast_nodes), " ".join(tokens)
            
    except javalang.parser.JavaSyntaxError as e:
        # Code uses syntax newer than Java 8 or has compilation issues
        pass 
    except Exception as e:
        pass
        
    return None, None

def build_file_index(root_folder):
    """
    Scans the directory tree EXACTLY ONCE and builds a lookup map.
    Key: base class filename (e.g., 'Main.java') -> Value: Full absolute/relative path
    """
    file_index = {}
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".java"):
                # If duplicate filenames exist in different packages, store a list or match via packages
                if file not in file_index:
                    file_index[file] = []
                file_index[file].append(os.path.join(root, file))
    return file_index

def locate_file_in_index(classname, file_index):
    # Handle inner classes (e.g., org.apache.camel.Main$1 -> Main.java)
    base_file_name = classname.split('$')[0].split('.')[-1] + ".java"
    
    paths = file_index.get(base_file_name, [])
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
        
    # Edge case: If multiple classes share the same filename across different packages,
    # match using package naming conventions
    package_path = classname.replace('.', os.sep).split('$')[0] + ".java"
    for path in paths:
        if package_path in path:
            return path
    return paths[0] # Default fallback

def run_extraction(project_name, versions):
    for v in versions:
        base_path = os.path.join("data", project_name)
        csv_input = os.path.join(base_path, f"{project_name}-{v}.csv")
        src_folder = os.path.join(base_path, f"src_{v}")
        csv_output = os.path.join(base_path, f"{project_name}_{v}_enriched.csv")

        if not os.path.exists(csv_input):
            print(f"⚠️ Skipping {v}: {csv_input} not found.")
            continue

        print(f"\n--- 📂 Deep Processing {project_name.upper()} Version {v} ---")
        df = pd.read_csv(csv_input)
        
        # Build index once!
        print(f"Building index for {src_folder}...")
        file_index = build_file_index(src_folder)
        
        all_ast = []
        all_tokens = []
        
        print(f"Processing {len(df)} records via Index Lookup...")
        for index, row in df.iterrows():
            classname = row.iloc[0]
            actual_path = locate_file_in_index(classname, file_index)
            
            ast, tokens = None, None
            if actual_path:
                ast, tokens = extract_java_features(actual_path)
            
            all_ast.append(ast)
            all_tokens.append(tokens)

        df['ast_seq'] = all_ast
        df['code_tokens'] = all_tokens

        df_cleaned = df.dropna(subset=['ast_seq', 'code_tokens'])
        df_cleaned = df_cleaned[df_cleaned['ast_seq'] != ""]

        print(f"✅ Success: {len(df_cleaned)}/{len(df)} files successfully enriched.")
        df_cleaned.to_csv(csv_output, index=False, encoding="utf-8", errors="replace")

if __name__ == "__main__":
    run_extraction("xerces", ["1.2", "1.3"])