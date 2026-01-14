import re
import ast

def parse_llm_res(res):
    paths = re.findall(r'\[\s*\[\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])*\s*\]', res)
    if paths:   
        soln = ast.literal_eval(paths[-1])
        return soln
    
    return []