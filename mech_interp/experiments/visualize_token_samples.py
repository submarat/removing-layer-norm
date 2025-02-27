from transformers import GPT2TokenizerFast
import html

def visualize_highlighted_tokens_in_markdown(tokens_list, highlight_idxs, tokenizer='gpt2', save_path=None, verbose=False, title=''):
    """
    Creates a markdown file with multiple sequences of tokens where the tokens at highlight_idxs are highlighted with different colors.
    Preserves the exact formatting of the original text.
    
    Args:
        tokens_list: List of lists of tokens, where each inner list represents a sequence
        highlight_idxs: List of indices to highlight in each sequence
        tokenizer: The tokenizer to decode tokens
        save_path: Path to save the markdown file
        verbose: If True, print additional information and include more details in the output
    """
    markdown_content = f'## {title}\n\n'
    if tokenizer == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer}. Only 'gpt2' is supported.")

    # Process each sequence
    for seq_idx, (tokens, highlight_idx) in enumerate(zip(tokens_list, highlight_idxs)):
        # Decode the full sequence in one go
        full_text = tokenizer.decode(tokens, skip_special_tokens=False)
        # full_text = ftfy.fix_text(full_text)
        full_text_pre_highlight = html.escape(tokenizer.decode(tokens[:highlight_idx], skip_special_tokens=False))
        full_text_post_highlight = html.escape(tokenizer.decode(tokens[highlight_idx+1:], skip_special_tokens=False))
        highlighted_token = html.escape(tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False))
        if highlighted_token.startswith('Ġ'):
            highlighted_token = highlighted_token[1:]

        following_token = ""
        if highlight_idx + 1 < len(tokens):
            following_token = html.escape(tokenizer.decode([tokens[highlight_idx+1]], skip_special_tokens=False))
            # Clean up the following token for display
            if following_token.startswith('Ġ'):
                following_token = ' ' + following_token[1:]
        
        # Build the highlighted text by decoding each token individually
        highlighted_text = full_text_pre_highlight + f"<span style='background-color:red'>{highlighted_token}</span>" + full_text_post_highlight

        # Add sequence header and information
        markdown_content += f"**Sequence {seq_idx + 1}**\n\n"
        
        if verbose:
            # Add the full text without highlighting first
            markdown_content += f"**Full text:**\n\n{full_text}\n\n"
            # Add the text with highlighting with title
            markdown_content += f"**With highlighting:**\n\n{highlighted_text}\n\n"
            
            if 0 <= highlight_idx < len(tokens):
                highlighted_token = tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False)
                # Clean up the token representation for display
                if highlighted_token.startswith('Ġ'):
                    highlighted_token = highlighted_token[1:]  # Remove the Ġ character for display
                
                markdown_content += f"**Highlighted token**: `{highlighted_token}`, **following_token**: `{following_token}`\n\n"
        else:
            # Just add the highlighted text without titles
            markdown_content += f"{highlighted_text}\n\n"
            markdown_content += f"**Highlighted token**: `{highlighted_token}`, **following_token**: `{following_token}`\n\n"
        

        # Add separator between sequences
        if seq_idx < len(tokens_list) - 1:  # Don't add separator after the last sequence
            markdown_content += "---\n\n"

    # Save markdown file
    if save_path:
        if not str(save_path).endswith('.md'):
            save_path = str(save_path).replace('.png', '.md')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Markdown visualization saved to {save_path}")
    return markdown_content