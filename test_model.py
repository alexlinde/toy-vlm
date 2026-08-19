"""
Interactive execution module for the Toy VLM.
Loads a trained model and allows interactive Q&A with generated shapes.
"""

import numpy as np
import os
import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import threading

from shapes import ShapeGenerator
from model import (
    load_trained_model,
    generate_response_traced,
    shape_probe_probabilities,
    model_shape_beliefs,
)
from device import DEVICE

def get_model_stats(model):
    """Get comprehensive model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Component-wise parameter counts
    vision_params = sum(p.numel() for p in model.vision_encoder.parameters())
    text_embed_params = sum(p.numel() for p in model.token_embedding.parameters()) + \
                      sum(p.numel() for p in model.position_embedding.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_blocks.parameters())
    output_params = sum(p.numel() for p in model.output_projection.parameters())

    # Model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'vision_params': vision_params,
        'text_embed_params': text_embed_params,
        'transformer_params': transformer_params,
        'output_params': output_params,
        'vocab_size': model.output_projection.out_features,
        'hidden_dim': model.token_embedding.embedding_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': model.transformer_blocks[0].self_attention.num_heads,
        'device': str(next(model.parameters()).device),
        'model_size_mb': model_size_mb
    }

def format_number(num):
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

class ToyVLMGUI:
    """Tkinter GUI for the Toy VLM."""
    
    def __init__(self, model_path='toy_vlm.pth'):
        # Initialize model + tokenizer + shape probe via the shared loader (the
        # checkpoint bundles its vocab so they can never mismatch). The probe is
        # optional: checkpoints trained before it existed yield None.
        self.model, self.tokenizer, self.probe = load_trained_model(model_path)
        self.text_processor = self.model.text_processor
        self.model.to(DEVICE)

        self.shape_generator = ShapeGenerator()

        self.current_shape_type = None
        self.current_image = None

        # Question history for navigation
        self.question_history = []
        self.history_index = -1

        # Image editing state
        self.editing_mode = 'square'
        self.erase_mode = False
        self.tool_size = 10
        self.canvas_scale = 300
        self.is_drawing = False

        # Introspection state: an (8, 8) patch-grid attention map, or None
        # when there is nothing to overlay.
        self.attention_map = None
        self._message_counter = 0

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Toy Vision-Language Model")
        self.root.geometry("800x720")
        self.setup_gui()
        
        # Generate initial shape
        self.generate_new_shape()
    
    def setup_gui(self):
        """Set up the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for image
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Image display (using Canvas for editing)
        self.canvas = tk.Canvas(left_frame, width=self.canvas_scale, height=self.canvas_scale, bg='black', highlightthickness=1)
        self.canvas.pack(pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # # Tool buttons
        edit_frame = ttk.Frame(left_frame)
        edit_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Shape selection radio buttons
        shapes_frame = ttk.Frame(edit_frame)        
        shapes_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.tool_var = tk.StringVar(value='square')
        ttk.Radiobutton(shapes_frame, text="Square", variable=self.tool_var, 
                       value='square', command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(shapes_frame, text="Circle", variable=self.tool_var, 
                       value='circle', command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        self.erase_var = tk.BooleanVar()
        ttk.Checkbutton(shapes_frame, text="Erase Mode", variable=self.erase_var, 
                       command=self.on_erase_change).pack(side=tk.RIGHT, padx=5)
        
        # Size slider
        size_frame = ttk.Frame(edit_frame)
        size_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.size_var = tk.IntVar(value=10)
        self.size_slider = ttk.Scale(size_frame, from_=5, to=30, orient=tk.HORIZONTAL, 
                                    variable=self.size_var, command=self.on_size_change)
        self.size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.size_label = ttk.Label(size_frame, text="10")
        self.size_label.pack(side=tk.LEFT)
        
        # Generate new shape button
        ttk.Button(edit_frame, text="New Shape", command=self.generate_new_shape).pack(pady=5, side=tk.LEFT)

        # Attention overlay toggle
        self.show_attention_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(edit_frame, text="Show attention", variable=self.show_attention_var,
                        command=self.update_canvas_display).pack(pady=5, side=tk.LEFT, padx=10)

        # Live shape beliefs: the model's own (language head) next to what the
        # vision tower alone can tell (linear probe on patch embeddings)
        self.belief_canvas = tk.Canvas(left_frame, width=self.canvas_scale, height=160,
                                       bg='white', highlightthickness=0)
        self.belief_canvas.pack(fill=tk.X, pady=(5, 0))

        # Attention inspector readout
        self.inspector_label = ttk.Label(
            left_frame, text="ask a question to see attention",
            wraplength=self.canvas_scale, font=('TkDefaultFont', 9), justify=tk.LEFT,
        )
        self.inspector_label.pack(fill=tk.X, pady=(4, 0))

        # Right panel for chat
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat history
        self.chat_display = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Question input
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(input_frame, text="Ask a question:").pack(anchor='w')

        # Entry and button in the same row
        entry_button_frame = ttk.Frame(input_frame)
        entry_button_frame.pack(fill=tk.X, pady=(5, 10))

        self.question_entry = ttk.Entry(entry_button_frame)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.question_entry.bind('<Return>', self.on_enter_pressed)
        self.question_entry.bind('<Up>', self.on_up_key)
        self.question_entry.bind('<Down>', self.on_down_key)

        # Send button
        ttk.Button(entry_button_frame, text="Ask Question", command=self.ask_question).pack(side=tk.RIGHT)
        
        # Display model statistics
        self.display_model_stats()

        # Add initial welcome message
        self.add_to_chat("Ask me what I can see in the image", "System")

        # Give focus to question entry
        self.question_entry.focus_set()
    
    def display_model_stats(self):
        """Display model statistics in the chat."""
        stats = get_model_stats(self.model)

        stats_text = f"📊 Model Statistics:\n"
        stats_text += f"• Total Parameters: {format_number(stats['total_params'])}\n"
        stats_text += f"• Vision Encoder: {format_number(stats['vision_params'])}\n"
        stats_text += f"• Text Embeddings: {format_number(stats['text_embed_params'])}\n"
        stats_text += f"• Transformer Blocks: {format_number(stats['transformer_params'])}\n"
        stats_text += f"• Output Layer: {format_number(stats['output_params'])}\n"
        stats_text += f"• Model Size: {stats['model_size_mb']:.1f} MB\n"
        stats_text += f"• Architecture: {stats['hidden_dim']}d, {stats['num_layers']} layers, {stats['num_heads']} heads\n"
        stats_text += f"• Vocabulary Size: {format_number(stats['vocab_size'])}\n"
        stats_text += f"• Device: {stats['device']}"

        self.add_to_chat(stats_text, "System")

    def add_to_chat(self, message, sender="User"):
        """Add a message to the chat display."""
        self.chat_display.config(state='normal')
        if sender == "System":
            self.chat_display.insert(tk.END, f"🤖 {message}\n\n")
        elif sender == "User":
            self.chat_display.insert(tk.END, f"👤 {message}\n")
        else:  # VLM response
            self.chat_display.insert(tk.END, f"🎯 {message}\n\n")

        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def generate_new_shape(self):
        """Generate a new random shape and update the display."""
        self.current_shape_type, self.current_image = self.shape_generator.generate_random_shape(add_noise=False)
        # Any attention map refers to the image being replaced.
        self.attention_map = None
        self.update_canvas_display()
        self.add_to_chat(f"Generated a new {self.current_shape_type}!", "System")
        self.update_shape_beliefs()

    def update_canvas_display(self):
        """Update the canvas with the current image (plus attention overlay)."""
        # Convert numpy array to PIL Image and then to PhotoImage
        img_array = (self.current_image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        self.img_size = pil_img.size

        if self.attention_map is not None and self.show_attention_var.get():
            pil_img = self._blend_attention(pil_img, self.attention_map)

        pil_img = pil_img.resize((self.canvas_scale, self.canvas_scale), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.photo, anchor='center')

    def _blend_attention(self, pil_img, attention_map):
        """Blend an (8, 8) attention map over the grayscale image as red heat."""
        peak = float(attention_map.max())
        if peak <= 0:
            return pil_img.convert('RGB')

        # Upsample the patch grid to image resolution
        heat_img = Image.fromarray(
            ((attention_map / peak) * 255).astype(np.uint8), mode='L'
        ).resize(pil_img.size, Image.BILINEAR)
        heat = np.asarray(heat_img, dtype=np.float32) / 255.0

        # Lerp each pixel toward red; the shape stays visible underneath
        base = np.asarray(pil_img.convert('RGB'), dtype=np.float32)
        alpha = (0.55 * heat)[..., None]
        red = np.array([255.0, 40.0, 40.0], dtype=np.float32)
        blended = base * (1.0 - alpha) + red * alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode='RGB')

    def update_shape_beliefs(self):
        """Redraw shape beliefs: the model's own (from its language head) as
        blue bars, paired with the vision-tower linear probe as green bars.
        The gap between the two is the point -- shape recognition happens in
        cross-attention, not in the vision encoder."""
        self.belief_canvas.delete("all")

        shapes = self.shape_generator.get_available_shapes()
        model_beliefs = model_shape_beliefs(self.model, self.current_image, shapes)
        probe_beliefs = (
            shape_probe_probabilities(self.model, self.probe, self.current_image)
            if self.probe is not None else None
        )

        # Legend
        self.belief_canvas.create_rectangle(8, 6, 18, 14, fill='#1565c0', outline='')
        self.belief_canvas.create_text(22, 10, anchor='w', text="model",
                                       font=('TkDefaultFont', 8))
        self.belief_canvas.create_rectangle(70, 6, 80, 14, fill='#2e7d32', outline='')
        self.belief_canvas.create_text(84, 10, anchor='w', text="vision tower",
                                       font=('TkDefaultFont', 8))
        if probe_beliefs is None:
            self.belief_canvas.create_text(
                160, 10, anchor='w', fill='#777777', font=('TkDefaultFont', 8),
                text="(no probe in checkpoint)",
            )

        best_model = max(model_beliefs, key=model_beliefs.get)
        best_probe = max(probe_beliefs, key=probe_beliefs.get) if probe_beliefs else None

        label_right = 70          # right edge of the fixed name column
        bar_left = label_right + 6
        bar_max = self.canvas_scale - bar_left - 44
        for row, name in enumerate(shapes):
            y = 32 + row * 26
            self.belief_canvas.create_text(label_right, y, anchor='e', text=name,
                                           font=('TkDefaultFont', 9))

            prob = model_beliefs[name]
            width = max(1, int(round(bar_max * prob)))
            fill = '#1565c0' if name == best_model else '#90caf9'
            self.belief_canvas.create_rectangle(bar_left, y - 9, bar_left + width, y - 2,
                                                fill=fill, outline='')
            self.belief_canvas.create_text(bar_left + width + 5, y - 5, anchor='w',
                                           text=f"{100 * prob:.0f}%",
                                           font=('TkDefaultFont', 7))

            if probe_beliefs is not None:
                prob = probe_beliefs[name]
                width = max(1, int(round(bar_max * prob)))
                fill = '#2e7d32' if name == best_probe else '#a5d6a7'
                self.belief_canvas.create_rectangle(bar_left, y + 2, bar_left + width, y + 9,
                                                    fill=fill, outline='')
                self.belief_canvas.create_text(bar_left + width + 5, y + 5, anchor='w',
                                               text=f"{100 * prob:.0f}%",
                                               font=('TkDefaultFont', 7))

    def on_enter_pressed(self, event):
        """Handle Enter key press in question entry."""
        self.ask_question()
    
    def on_up_key(self, event):
        """Handle Up arrow key press - navigate to previous question in history."""
        if not self.question_history:
            return
        
        # If at end of history, move to last item
        if self.history_index == -1:
            self.history_index = len(self.question_history) - 1
        # Otherwise move backwards
        elif self.history_index > 0:
            self.history_index -= 1
        
        # Load the question at current index
        if 0 <= self.history_index < len(self.question_history):
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])
    
    def on_down_key(self, event):
        """Handle Down arrow key press - navigate to next question in history."""
        if not self.question_history or self.history_index == -1:
            return
        
        # Move forward in history
        if self.history_index < len(self.question_history) - 1:
            self.history_index += 1
            self.question_entry.delete(0, tk.END)
            self.question_entry.insert(0, self.question_history[self.history_index])
        else:
            # At end of history, clear entry and reset index
            self.history_index = -1
            self.question_entry.delete(0, tk.END)
    
    def ask_question(self):
        """Process a question about the current shape."""
        question = self.question_entry.get().strip()
        if not question:
            return
        
        # Clear the input
        self.question_entry.delete(0, tk.END)
        
        # Add question to history and reset history index
        self.question_history.append(question)
        self.history_index = -1
        
        # Add question to chat
        self.add_to_chat(question, "User")

        # Warn about words the model was never trained on -- it will treat
        # them as untrained <UNK> noise rather than genuinely understanding them.
        oov = self.tokenizer.oov_words(question)
        if oov:
            self.add_to_chat(
                f"Note: not in the model's vocabulary: {', '.join(oov)} — it cannot understand these words.",
                "System",
            )

        # Process in background thread to avoid freezing GUI. Snapshot the
        # current image now so a later New Shape / draw doesn't change which
        # image the answer is about.
        threading.Thread(
            target=self._process_question,
            args=(question, self.current_image.copy()),
            daemon=True,
        ).start()

    def _process_question(self, question, image):
        """Process the question in a background thread."""
        try:
            response, trace = generate_response_traced(self.model, image, question)
        except Exception as e:
            self.root.after(0, self.add_to_chat, f"Error: {e}", "System")
            return

        # Update GUI in main thread
        self.root.after(0, self._deliver_response, image, response, trace)

    @staticmethod
    def _confidence_color(prob):
        """Chat background colour for a token's confidence."""
        if prob >= 0.9:
            return '#c8e6c9'  # green
        if prob >= 0.6:
            return '#ffe0b2'  # amber
        return '#ffcdd2'      # red

    def _deliver_response(self, image, response, trace):
        """Render the answer in the chat, colour-coded by per-token confidence."""
        if not trace:
            # Empty generation: nothing to colour or inspect.
            self.add_to_chat(response, "VLM")
        else:
            self._message_counter += 1
            msg = self._message_counter

            self.chat_display.config(state='normal')
            self.chat_display.insert(tk.END, "🎯 ")
            for i, entry in enumerate(trace):
                tag = f"tok{msg}_{i}"
                self.chat_display.insert(tk.END, entry['word'], tag)
                self.chat_display.insert(tk.END, " ")
                self.chat_display.tag_config(tag, background=self._confidence_color(entry['prob']))
                # Bind this word to its own trace and image snapshot, so words
                # in older answers keep inspecting their own tokens after newer
                # answers arrive.
                self.chat_display.tag_bind(
                    tag, "<Button-1>",
                    lambda event, tr=trace, img=image, idx=i: self._inspect_token(tr, img, idx)
                )
            self.chat_display.insert(tk.END, "\n")

            # Dim per-token confidence line, with the runner-up where it matters
            parts = []
            for entry in trace:
                part = f"{entry['word']} {100 * entry['prob']:.0f}%"
                alternatives = [(w, p) for w, p in entry['top_k'] if w != entry['word']]
                if alternatives and alternatives[0][1] >= 0.05:
                    part += f" ({alternatives[0][0]} {100 * alternatives[0][1]:.0f}%)"
                parts.append(part)

            conf_tag = f"conf{msg}"
            self.chat_display.insert(tk.END, ' · '.join(parts) + "\n\n", conf_tag)
            self.chat_display.tag_config(conf_tag, foreground='#777777',
                                         font=('TkDefaultFont', 8))

            self.chat_display.config(state='disabled')
            self.chat_display.see(tk.END)

        # Only overlay when the answer is still about what's on screen: the user
        # may have drawn or generated a new shape while inference was running.
        if trace and np.array_equal(image, self.current_image):
            self.attention_map = np.mean([entry['attention'] for entry in trace], axis=0)
            self.update_canvas_display()
            self.inspector_label.config(
                text="attention: answer average — click an answer word to inspect a token"
            )

    def _inspect_token(self, trace, image, i):
        """Show the attention map and alternatives for one answer token."""
        if not np.array_equal(image, self.current_image):
            self.inspector_label.config(
                text="image changed — attention for this answer no longer applies"
            )
            return

        entry = trace[i]
        self.attention_map = entry['attention']

        text = f"'{entry['word']}' {100 * entry['prob']:.0f}%"
        alternatives = [(w, p) for w, p in entry['top_k'] if w != entry['word']]
        if alternatives:
            text += " · alternatives: " + ", ".join(
                f"{w} {100 * p:.0f}%" for w, p in alternatives
            )
        self.inspector_label.config(text=text)
        self.update_canvas_display()


    def on_tool_change(self):
        """Handle tool selection change."""
        self.editing_mode = self.tool_var.get()
    
    def on_erase_change(self):
        """Handle erase mode checkbox change."""
        self.erase_mode = self.erase_var.get()
    
    def on_size_change(self, value):
        """Handle size slider change."""
        self.tool_size = int(float(value))
        self.size_label.config(text=str(self.tool_size))
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas."""
        self.is_drawing = True
        self.draw_at_position(event.x, event.y)
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas."""
        if self.is_drawing:
            self.draw_at_position(event.x, event.y)
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas."""
        _ = event  # Unused parameter
        self.is_drawing = False
        # The image changed under the probe; refresh its beliefs once the
        # interaction ends rather than on every drag event.
        self.update_shape_beliefs()

    def draw_at_position(self, canvas_x, canvas_y):
        """Draw at the specified canvas position."""
        # Convert canvas coordinates to image coordinates (300x300 -> 64x64)
        img_x = int(canvas_x * self.img_size[0] / self.canvas_scale)
        img_y = int(canvas_y * self.img_size[1] / self.canvas_scale)
        
        # Ensure coordinates are within bounds
        if 0 <= img_x < self.img_size[0] and 0 <= img_y < self.img_size[1]:
            self.draw_shape(self.editing_mode, img_x, img_y, self.tool_size, 0 if self.erase_mode else 255)
            self.update_canvas_display()
    
    def draw_shape(self, shape_type, center_x, center_y, size, fill_color):
        """Draw or erase a shape at the specified position using Pillow."""
        # Any attention map refers to the image before this edit.
        self.attention_map = None

        # Convert numpy array to PIL Image
        pil_img = Image.fromarray((self.current_image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        half_size = size // 2
        x1 = center_x - half_size; y1 = center_y - half_size;
        x2 = center_x + half_size; y2 = center_y + half_size;
        
        if shape_type == 'square':
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)            
        elif shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=fill_color)
        
        # Convert back to numpy array
        self.current_image = np.array(pil_img, dtype=np.float32) / 255.0
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='toy_vlm.pth', help='Path to trained model checkpoint (.pth)')
    args = parser.parse_args()

    try:
        gui = ToyVLMGUI(model_path=args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    gui.run()

if __name__ == "__main__":
    main()