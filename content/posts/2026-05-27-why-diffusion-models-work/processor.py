import argparse
import os

def wrap_latex_with_raw(text):
    result = []
    stack = []
    i = 0
    inside_block = False  # Track if we are inside a block LaTeX environment
    inside_heading = False
    while i < len(text):
        if text[i:i+2] == '$$' and text[i-1] != "\\":  # Detect block LaTeX start or end
            if inside_block:
                # End of block LaTeX
                start = stack.pop()
                content = text[start+2:i]

                result.append(f"{{{{< rawhtml >}}}}\n$$\n{content}\n$$\n{{{{< /rawhtml >}}}}")
                inside_block = False
            else:
                # Start of block LaTeX
                stack.append(i)
                inside_block = True
            i += 1  # Skip the next '$'
        elif text[i] == '$' and text[i-1] != "\\" and not inside_block and not inside_heading:  # Detect inline LaTeX start or end
            if stack and text[stack[-1]] == '$':
                # End of inline LaTeX
                start = stack.pop()
                content = text[start+1:i]

                result.append(f"{{{{< rawhtml >}}}}${content}${{{{< /rawhtml >}}}}")
                
            else:
                # Start of inline LaTeX
                stack.append(i)
        elif not inside_block and not stack:
            # If not within any LaTeX block or inline, add the character to the result
            if text[i] == '#':
                inside_heading = True
            if inside_heading and text[i] == '\n':
                inside_heading = False
            result.append(text[i])
        i += 1

    # Append remaining text after the last $$ or $ if any unprocessed
    while stack:
        start = stack.pop()
        result.append(text[start:])

    return ''.join(result)

def remove_initial_headings(text):
    """
    Remove the initial lines starting with '#' from the given text until a line not starting with '#' is found.

    Args:
    text (str): The input text from which the lines starting with '#' should be removed.

    Returns:
    str: The text after removing the initial '#' lines.
    """
    lines = text.splitlines()
    index = 0
    # Find the index of the first line that does not start with '#'
    
    while index < len(lines) and (lines[index].startswith('#') or lines[index].isspace() or lines[index] == ""):
        index += 1
    
    # Join the remaining lines back into a single string
    return '\n'.join(lines[index:])

def process_file(input_file):
    # Generate the output file name with .md extension
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.md"

    # Read input file
    with open(input_file, 'r') as file:
        content = file.read()

    # Process the content
    processed_content = wrap_latex_with_raw(content)
    processed_content = remove_initial_headings(processed_content)
    # Write output file
    with open(output_file, 'w') as file:
        file.write(processed_content)

    print(f"Processed content written to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process LaTeX in a file and wrap it with rawhtml tags.")
    parser.add_argument("filename", help="The input file to process.")

    # Parse the command line arguments
    args = parser.parse_args()

    # Process the file
    process_file(args.filename)
