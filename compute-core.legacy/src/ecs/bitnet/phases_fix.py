import re

with open("compute-core/src/bitnet/phases.rs") as f:
    content = f.read()

# Replace each emit_checkpoint_ternary_tensor call that has 10 args (missing group_size)
# Pattern: look for the call with exactly 10 closing params before `)?;`
# We add `group_size,` before the `)?;`

# Strategy: find all occurrences and add group_size where missing
count = 0
lines = content.split('\n')
result = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    if stripped.startswith("emit_checkpoint_ternary_tensor("):
        # Scan forward for the matching closing
        paren_depth = 1
        start_i = i
        i += 1
        while i < len(lines) and paren_depth > 0:
            for ch in lines[i]:
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
            if paren_depth > 0:
                i += 1
        
        end_i = i
        
        # Now we have the full call from start_i to end_i
        call_text = '\n'.join(lines[start_i:end_i+1])
        
        # Check if group_size is already passed
        if 'group_size,' not in call_text:
            # Add group_size before the closing )?;
            # Find the last )?; line
            last_line = lines[end_i].strip()
            if last_line.endswith(')?;'):
                indent = lines[end_i][:len(lines[end_i]) - len(lines[end_i].lstrip())]
                lines[end_i] = indent + '            group_size,'
                end_i += 1
                lines.insert(end_i, f"{indent}        )?;")
            count += 1
        
        i = end_i + 1
    else:
        i += 1

result = '\n'.join(lines)
with open("compute-core/src/bitnet/phases.rs", "w") as f:
    f.write(result)

print(f"Fixed {count} call sites")
