# 🚀 Deploying ArchitectAI to Blaxcel AI

## Prerequisites

1. **Blaxcel AI Account**: Sign up at https://app.blaxcel.ai/
2. **Python 3.10+**: Already have it
3. **MCP Package**: Install `mcp` library

## Step 1: Install Dependencies

```bash
# Install MCP library
pip install mcp

# Or install from PyPI (once published)
pip install architectai-mcp
```

## Step 2: Test the MCP Server Locally

```bash
# Test the MCP server
python mcp_server.py

# You should see:
# 🚀 Starting ArchitectAI MCP Server
# Available tools:
#   - analyze_patterns: Analyze Python code for design patterns...
#   - analyze_project: Analyze entire project structure...
#   - etc.
```

## Step 3: Register on Blaxcel AI

1. Go to https://app.blaxcel.ai/
2. Sign up for an account
3. Get your API credentials
4. Create a new MCP integration

## Step 4: Prepare for Deployment

### Update mcp.json
Replace placeholders:
```bash
sed -i 's|yourusername|YOUR_GITHUB_USERNAME|g' mcp.json
sed -i 's|team@architectai.dev|YOUR_EMAIL|g' mcp.json
```

### Add Environment Variables
Create `.env.blaxcel`:
```bash
NEBIUS_API_KEY=your_nebius_key
OPENAI_API_KEY=your_openai_key (optional)
SAMBANOVA_API_KEY=your_sambanova_key (optional)
BLAXCEL_AUTH_TOKEN=your_blaxcel_token
```

## Step 5: Push to Blaxcel AI

### Using Blaxcel CLI (Recommended)

```bash
# Install Blaxcel CLI
pip install blaxcel-cli

# Login to Blaxcel
blaxcel login --token YOUR_BLAXCEL_TOKEN

# Push the MCP
blaxcel push \
  --name "architectai-mcp" \
  --description "Design Pattern Analysis & AI Refactoring" \
  --version "1.0.0" \
  --entry-point "mcp_server.py" \
  --config "mcp.json" \
  --env-file ".env.blaxcel"

# Verify deployment
blaxcel list

# Test the MCP
blaxcel test architectai-mcp
```

### Manual Upload (Alternative)

1. Go to https://app.blaxcel.ai/dashboard
2. Click "Add MCP"
3. Upload `mcp.json`
4. Upload `mcp_server.py`
5. Set environment variables in dashboard
6. Click "Deploy"

## Step 6: Use in Claude/AI Assistants

Once deployed, you can use the MCP in any compatible AI assistant:

```
User: Analyze this code for design patterns
[pastes Python code]

Claude: I'm using the ArchitectAI MCP to analyze your code...

(Uses analyze_patterns tool)

Claude: I found 3 design patterns:
1. Singleton - Database class
2. Factory - ProductFactory
3. Observer - EventSystem
```

## Step 7: Share with Community

### Publish to Blaxcel Marketplace

1. Your MCP is automatically available to Blaxcel users
2. Share the marketplace link: `https://app.blaxcel.ai/mcp/architectai-mcp`
3. GitHub badge:
```markdown
[![Blaxcel MCP](https://img.shields.io/badge/Blaxcel-MCP-blue)](https://app.blaxcel.ai/mcp/architectai-mcp)
```

### Publish to PyPI (Optional)

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install architectai-mcp
```

## Troubleshooting

### Error: "ModuleNotFoundError: mcp"
```bash
pip install mcp --upgrade
```

### Error: "API Key Invalid"
```bash
# Check environment variables
echo $NEBIUS_API_KEY

# Re-authenticate with Blaxcel
blaxcel login --token YOUR_TOKEN --force
```

### Error: "Tool not found"
```bash
# Verify mcp_server.py has correct imports
python -c "from mcp_server import *"

# Check mcp.json is valid JSON
python -m json.tool mcp.json
```

## Configuration

### mcp.json Structure
- `name`: Unique identifier
- `version`: Semantic versioning
- `capabilities.tools`: List of MCP tools
- `requirements`: Python dependencies
- `environment`: Required env variables

### Available Tools

1. **analyze_patterns**
   - Input: Python code
   - Output: Pattern analysis report
   - Enrichment: LLM-powered explanations

2. **analyze_project**
   - Input: ZIP file path
   - Output: Architecture diagrams
   - Scope: Entire project structure

3. **detect_patterns**
   - Input: Code + structure
   - Output: Pattern detection with UML
   - Features: Before/after diagrams

4. **get_refactoring_proposal**
   - Input: Code + refactoring instruction
   - Output: AI-powered suggestions
   - LLM providers: Nebius, OpenAI, SambaNova

5. **generate_uml**
   - Input: Python code
   - Output: PlantUML diagram
   - Options: Include/exclude methods

## Monitoring & Analytics

After deployment, monitor usage:
1. Go to Blaxcel Dashboard
2. Check "architectai-mcp" MCP stats
3. View:
   - Total calls per tool
   - Average response time
   - Error rate
   - Usage by geography

## Updating Your MCP

To push updates:

```bash
# Increment version in mcp.json
# Update code as needed
# Run tests
pytest tests/

# Push update
blaxcel push --update
```

## Support & Documentation

- **Docs**: https://github.com/yourusername/architectai-mcp/wiki
- **Issues**: https://github.com/yourusername/architectai-mcp/issues
- **Blaxcel Support**: https://support.blaxcel.ai
- **Discord**: https://discord.gg/blaxcel

## Next Steps

1. ✅ Deploy MCP to Blaxcel
2. 📊 Monitor usage and metrics
3. 🚀 Iterate based on user feedback
4. 🌟 Get featured in Blaxcel marketplace
5. 💰 Monetize (if applicable)

---

**Questions?** Open an issue or contact support@architectai.dev
