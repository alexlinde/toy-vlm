"""
Question generation module for the Toy VLM.
Handles creation of questions and answers about geometric shapes using Jinja templates.
"""

import random
from typing import Tuple
from jinja2 import Environment, BaseLoader
from shapes import ShapeGenerator

class TemplateLoader(BaseLoader):
    """Custom Jinja2 template loader that loads templates from text file."""
    
    def __init__(self, templates_file: str):
        self.templates_file = templates_file
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from text file with pipe-separated format."""
        self.templates_data = {}
        with open(self.templates_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and '|' in line:
                    template_name = f"template_{i}"
                    self.templates_data[template_name] = line
    
    def get_source(self, environment, template):
        """Get template source for Jinja2."""
        if template in self.templates_data:
            source = self.templates_data[template]
            return source, None, lambda: True
        raise Exception(f"Template {template} not found")


class QuestionGenerator:
    """Generates questions about shapes using Jinja2 templates."""
    
    def __init__(self, templates_file: str = "questions.txt"):
        self.shape_generator = ShapeGenerator()
        self.shapes = self.shape_generator.get_available_shapes()
        
        # Set up Jinja2 environment
        self.template_loader = TemplateLoader(templates_file)
        self.env = Environment(loader=self.template_loader)
        
        # Load available template names
        self.template_names = list(self.template_loader.templates_data.keys())
    
    def generate_qa_pair(self, shape_type: str) -> Tuple[str, str]:
        """Generate a question-answer pair about the shape using templates."""
        # Select a random template
        template_name = random.choice(self.template_names)
        template = self.env.get_template(template_name)
        
        # Render the template with shape context
        other_shapes = [s for s in self.shapes if s != shape_type]
        context = {
            'shape': shape_type,
            'random_other_shape': random.choice(other_shapes)
        }
        
        result = template.render(**context)
        
        # Split into question and answer (templates should be formatted as "question|answer")
        if '|' in result:
            question, answer = result.split('|', 1)
            return question.strip(), answer.strip()
        else:
            # Fallback for templates that don't use the pipe separator
            return result.strip(), f"this is a {shape_type} ."
    