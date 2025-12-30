"""
Generate PDF documentation for the AI Nutrition project
This script creates a comprehensive PDF explaining each file's purpose
"""

from fpdf import FPDF

class PDFDocumentation(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(75, 0, 130)  # Indigo
        self.cell(0, 10, 'AI Nutrition Project - File Documentation', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(34, 139, 34)  # Forest green
        self.ln(5)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        
    def add_file_entry(self, filename, purpose):
        self.set_font('Courier', 'B', 10)
        self.set_text_color(0, 100, 150)  # Steel blue
        self.cell(0, 7, filename, new_x='LMARGIN', new_y='NEXT')
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, purpose)
        self.ln(3)

def create_documentation():
    pdf = PDFDocumentation()
    
    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(75, 0, 130)
    pdf.ln(40)
    pdf.cell(0, 15, 'AI Nutrition Project', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 18)
    pdf.cell(0, 12, 'Complete File Documentation', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'An intelligent nutrition analysis system using AI', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 8, 'with food database integration', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(30)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 8, 'Generated: December 2024', align='C', new_x='LMARGIN', new_y='NEXT')
    
    # Content pages
    pdf.add_page()
    
    # Root Configuration Files
    pdf.add_section_title('Root Configuration Files')
    
    root_files = [
        ('.env', 'Stores sensitive configuration variables like API keys for Groq AI and USDA FoodData Central. This file is never committed to version control for security.'),
        ('.env.example', 'Template showing required environment variables without actual values. Developers copy this to .env and fill in their own API keys.'),
        ('.dockerignore', 'Tells Docker which files to exclude when building container images (like venv, __pycache__, etc.) to keep images smaller.'),
        ('Dockerfile', 'Container build instructions for deploying the FastAPI application. Sets up Python environment and configures the Uvicorn server.'),
        ('docker-compose.yml', 'Multi-container orchestration configuration. Defines the web service with port mappings and environment variable handling.'),
        ('requirements.txt', 'Lists all Python dependencies (FastAPI, httpx, groq, cachetools, etc.) that pip needs to install.'),
        ('pytest.ini', 'Configuration for pytest testing framework. Sets up test discovery paths and markers.'),
        ('README.md', 'Project overview and getting started guide. Contains installation instructions and basic usage examples.'),
    ]
    
    for filename, purpose in root_files:
        pdf.add_file_entry(filename, purpose)
    
    # Source Code - Main Application
    pdf.add_page()
    pdf.add_section_title('Source Code - Main Application (src/)')
    
    main_files = [
        ('src/main.py', 'The heart of the application. Creates the FastAPI app, configures middleware for security, and sets up all API routes. Handles application startup and template rendering.'),
        ('src/__init__.py', 'Makes src a Python package. Often empty but essential for Python imports to work correctly across the project.'),
        ('src/config/__init__.py', 'Package initializer for configuration module. Exports key configuration objects for use throughout the app.'),
        ('src/config/settings.py', 'Loads and validates all configuration values from environment variables. Uses Pydantic for type validation and default values.'),
    ]
    
    for filename, purpose in main_files:
        pdf.add_file_entry(filename, purpose)
    
    # Data Models
    pdf.add_section_title('Data Models (src/models/)')
    
    model_files = [
        ('src/models/__init__.py', 'Exports all Pydantic models used for request/response validation and data serialization throughout the API.'),
        ('src/models/food_models.py', 'Defines data structures for food items - nutrients, serving sizes, and nutritional information from USDA database.'),
        ('src/models/analysis_models.py', 'Structures for AI analysis results - meal scores, recommendations, and dietary assessments.'),
        ('src/models/request_models.py', 'Input validation models for API requests - what data clients must send and in what format.'),
        ('src/models/response_models.py', 'Output structure models defining exactly what the API returns for each endpoint type.'),
    ]
    
    for filename, purpose in model_files:
        pdf.add_file_entry(filename, purpose)
    
    # API Routes
    pdf.add_page()
    pdf.add_section_title('API Routes (src/routes/)')
    
    route_files = [
        ('src/routes/__init__.py', 'Collects and exports all route modules. Makes including all routes in main.py simple with a single import.'),
        ('src/routes/food_routes.py', 'Endpoints for food search, lookup, and autocomplete. Connects frontend requests to USDA service.'),
        ('src/routes/analysis_routes.py', 'Endpoints for AI-powered meal analysis. Sends food data to Groq AI and returns nutritional insights.'),
        ('src/routes/health_routes.py', 'System health check endpoints. Used by Docker and monitoring tools to verify the app is running.'),
    ]
    
    for filename, purpose in route_files:
        pdf.add_file_entry(filename, purpose)
    
    # Core Services
    pdf.add_section_title('Core Services (src/services/)')
    
    service_files = [
        ('src/services/__init__.py', 'Exports service instances for dependency injection. Manages singleton service instances.'),
        ('src/services/groq_service.py', 'Communicates with Groq AI (Llama models) for intelligent meal analysis and nutritional recommendations.'),
        ('src/services/usda_service.py', 'Fetches food data from USDA FoodData Central API. Handles caching to reduce API calls.'),
        ('src/services/meal_analyzer.py', 'Orchestrates the meal analysis process. Combines USDA data with AI insights to generate comprehensive reports.'),
    ]
    
    for filename, purpose in service_files:
        pdf.add_file_entry(filename, purpose)
    
    # Intelligence Module
    pdf.add_page()
    pdf.add_section_title('AI Intelligence (src/intelligence/)')
    
    intel_files = [
        ('src/intelligence/__init__.py', 'Entry point for the AI module. Exports the main intelligence classes and functions.'),
        ('src/intelligence/base.py', 'Abstract base classes defining the interface for all AI analyzers. Ensures consistent behavior.'),
        ('src/intelligence/analyzers.py', 'Concrete analyzer implementations. Different strategies for analyzing various dietary aspects.'),
        ('src/intelligence/prompts.py', 'Carefully crafted prompts for the Llama AI. These prompts guide the AI to give nutritionally accurate responses.'),
        ('src/intelligence/fallback.py', 'Provides reasonable responses when AI service is unavailable. Ensures app never completely fails.'),
    ]
    
    for filename, purpose in intel_files:
        pdf.add_file_entry(filename, purpose)
    
    # Security Module
    pdf.add_section_title('Security (src/security/)')
    
    security_files = [
        ('src/security/__init__.py', 'Security module exports. Provides easy access to rate limiting and input validation.'),
        ('src/security/rate_limiter.py', 'Prevents API abuse by limiting request frequency. Protects against DoS attacks and API cost overruns.'),
        ('src/security/validators.py', 'Input sanitization and validation. Prevents injection attacks and ensures data integrity.'),
    ]
    
    for filename, purpose in security_files:
        pdf.add_file_entry(filename, purpose)
    
    # Cache Module
    pdf.add_section_title('Caching (src/cache/)')
    
    cache_files = [
        ('src/cache/__init__.py', 'Exports caching utilities for use across services.'),
        ('src/cache/cache_manager.py', 'Manages in-memory caching with TTL (time-to-live). Reduces USDA API calls and improves response times.'),
    ]
    
    for filename, purpose in cache_files:
        pdf.add_file_entry(filename, purpose)
    
    # Utilities
    pdf.add_page()
    pdf.add_section_title('Utilities (src/utils/)')
    
    util_files = [
        ('src/utils/__init__.py', 'Exports utility functions used across multiple modules.'),
        ('src/utils/helpers.py', 'General helper functions - string formatting, data transformation, nutritional calculations.'),
        ('src/utils/constants.py', 'Application-wide constants like nutrient RDA values, measurement conversions, and dietary reference intakes.'),
    ]
    
    for filename, purpose in util_files:
        pdf.add_file_entry(filename, purpose)
    
    # Templates
    pdf.add_section_title('Frontend Templates (src/templates/)')
    
    template_files = [
        ('src/templates/index.html', 'Main web interface. A modern, responsive single-page application for searching foods and viewing analysis.'),
        ('src/templates/base.html', 'Base HTML template with common elements (head, navigation, footer) inherited by other templates.'),
    ]
    
    for filename, purpose in template_files:
        pdf.add_file_entry(filename, purpose)
    
    # Static Files
    pdf.add_section_title('Static Assets (static/)')
    
    static_files = [
        ('static/css/styles.css', 'Main stylesheet with modern design - gradients, animations, responsive layouts, and dark theme support.'),
        ('static/js/app.js', 'Frontend JavaScript handling user interactions, API calls, and dynamic content rendering.'),
        ('static/js/charts.js', 'Visualization logic using Chart.js for nutritional data display (macros pie charts, nutrient bars).'),
    ]
    
    for filename, purpose in static_files:
        pdf.add_file_entry(filename, purpose)
    
    # Tests
    pdf.add_page()
    pdf.add_section_title('Test Suite (tests/)')
    
    test_files = [
        ('tests/__init__.py', 'Makes tests a Python package. Often contains shared test fixtures and configurations.'),
        ('tests/conftest.py', 'Pytest fixtures shared across all tests. Includes test client setup and mock data factories.'),
        ('tests/test_intelligence.py', 'Unit tests for the AI intelligence module. Verifies analyzer logic and prompt handling.'),
        ('tests/test_phase3.py', 'Integration tests for Phase 3 features. Tests the complete meal analysis pipeline.'),
    ]
    
    for filename, purpose in test_files:
        pdf.add_file_entry(filename, purpose)
    
    # Documentation
    pdf.add_section_title('Documentation (docs/)')
    
    doc_files = [
        ('docs/AI_LIMITATIONS.md', 'Explains what the AI can and cannot do. Sets realistic expectations for nutritional analysis accuracy.'),
        ('docs/API.md', 'Complete API reference with endpoint descriptions, request/response examples, and error codes.'),
        ('docs/ARCHITECTURE.md', 'System architecture documentation. Explains how components interact and data flows.'),
        ('docs/CONTRIBUTING.md', 'Guidelines for contributors. Code style, PR process, and development setup instructions.'),
    ]
    
    for filename, purpose in doc_files:
        pdf.add_file_entry(filename, purpose)
    
    # Data and Scripts
    pdf.add_section_title('Data & Scripts')
    
    other_files = [
        ('data/', 'Directory for local data files - cached responses, user data, and temporary storage.'),
        ('scripts/init_db.py', 'Database initialization script. Sets up tables and initial data if using persistent storage.'),
        ('models/', 'Directory potentially for ML models or data models separate from Pydantic request/response models.'),
    ]
    
    for filename, purpose in other_files:
        pdf.add_file_entry(filename, purpose)
    
    # Save the PDF
    output_path = r"c:\Users\hp\AI Nutrition\AI_Nutrition_File_Documentation.pdf"
    pdf.output(output_path)
    print(f"PDF created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_documentation()
