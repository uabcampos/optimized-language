# Smart PDF Language Flagger - Advanced Web Interface

A user-friendly web interface for the Smart PDF Language Flagger tool that allows you to upload documents, configure language flagging settings, and download results through an intuitive web UI.

## Features

### 🚀 Core Features
- **File Upload**: Upload PDF and DOCX files directly through the web interface
- **Configuration Management**: Edit flagged terms and replacements through JSON editors
- **Real-time Processing**: Process documents with progress tracking
- **Results Download**: Download annotated PDFs, CSV reports, and JSON data

### 📊 Advanced Features (Advanced Version)
- **Interactive Visualizations**: Charts and graphs showing flag distribution
- **Configuration Presets**: Pre-loaded configurations for different use cases
- **Batch Analysis**: Comprehensive statistics and analytics
- **ZIP Downloads**: Download all results in a single ZIP file

## Quick Start

### 1. Install Dependencies
```bash
# Install web interface dependencies
pip install -r requirements_web.txt
```

### 2. Start the Web Interface

#### Basic Version
```bash
streamlit run web_app.py
```

#### Advanced Version (with analytics)
```bash
streamlit run web_app_advanced.py
```

#### Using the Startup Script
```bash
./run_web_app.sh
```

### 3. Access the Interface
Open your browser and go to: `http://localhost:8501`

## Usage

### 1. Upload Document
- Click "Choose a PDF or DOCX file" in the sidebar
- Select your document to process

### 2. Configure Settings
- **LLM Settings**: Choose model and temperature
- **Processing Settings**: Select annotation style
- **Flagged Terms**: Edit the JSON array of terms to flag
- **Replacements**: Edit the JSON object of term replacements

### 3. Process Document
- Click "🚀 Process Document" to start processing
- Wait for the processing to complete

### 4. View and Download Results
- View detailed results in the table
- Download CSV reports, JSON data, and annotated PDFs
- (Advanced) View interactive visualizations and analytics

## Configuration Presets

The advanced version includes several configuration presets:

- **Grant (Enhanced)**: Uses `flagged_terms_grant_enhanced.json` and `replacements_grant_enhanced.json`
- **RPPR**: Uses `flagged_terms_grant.json` and `replacements_grant.json`
- **General**: Uses `flagged_terms.json` and `replacements.json`

## File Structure

```
web_app.py              # Basic web interface
web_app_advanced.py     # Advanced web interface with analytics
requirements_web.txt    # Web interface dependencies
run_web_app.sh         # Startup script
README_WEB.md          # This file
```

## Advanced Features

### Visualizations
- **Page Distribution**: Bar chart showing flags per page
- **Top Flagged Terms**: Horizontal bar chart of most common terms
- **Suggestion Analysis**: Word frequency analysis of suggestions

### Download Options
- **CSV Report**: Detailed spreadsheet of all flags
- **JSON Report**: Machine-readable JSON format
- **Annotated PDF**: PDF with highlighted flagged terms
- **ZIP Archive**: All files bundled together

### Statistics
- Total number of flags found
- Number of unique terms flagged
- Number of pages with flags
- Average flags per page

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Run: `pip install -r requirements_web.txt`

2. **OpenAI API errors**
   - Ensure your `.env` file contains a valid `OPENAI_API_KEY`

3. **File upload issues**
   - Check that the file is a valid PDF or DOCX
   - Ensure the file is not corrupted

4. **Processing fails**
   - Check the error message in the interface
   - Verify your JSON configuration is valid
   - Ensure the input file is accessible

### Getting Help

If you encounter issues:
1. Check the error messages in the web interface
2. Verify your configuration files are valid JSON
3. Ensure all dependencies are installed
4. Check that your OpenAI API key is valid

## Customization

### Adding New Presets
1. Create new JSON configuration files
2. Add the preset to the `preset` selectbox in the code
3. Update the file loading logic

### Modifying the Interface
- Edit `web_app.py` for basic changes
- Edit `web_app_advanced.py` for advanced features
- Customize the styling using Streamlit's theming options

## Security Notes

- The web interface runs locally on your machine
- Files are temporarily stored during processing
- No data is sent to external servers except for OpenAI API calls
- Temporary files are cleaned up after processing

## Performance Tips

- For large documents, processing may take several minutes
- Use the progress indicators to monitor processing status
- Consider using faster models (gpt-4o-mini) for quicker processing
- Close other applications to free up system resources
