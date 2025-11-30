# ArchitectAI MCP - Quick Reference

## 📦 Files Created

```
architectai-mcp/
├── mcp_server.py              # MCP Server implementation
├── mcp.json                   # MCP configuration
├── setup.py                   # PyPI setup
├── BLAXCEL_DEPLOYMENT.md      # Deployment guide
├── test_mcp.py               # Test suite
└── MCP_QUICKSTART.md         # This file
```

## 🚀 Quick Start

### 1. Test Locally
```bash
python test_mcp.py
```

### 2. Deploy to Blaxcel AI
```bash
# Install CLI
pip install blaxcel-cli

# Login
blaxcel login --token YOUR_TOKEN

# Push
blaxcel push --config mcp.json --entry-point mcp_server.py
```

## 🛠️ Available MCP Tools

### 1. `analyze_patterns`
Detect design patterns in Python code with AI recommendations.

**Usage:**
```json
{
  "code": "your python code",
  "enrich": true,
  "provider": "nebius"
}
```

**Returns:**
- Pattern detection report (markdown)
- Number of patterns found
- List of recommendations
- Confidence scores

### 2. `analyze_project`
Analyze entire project structure from ZIP file.

**Usage:**
```json
{
  "zip_path": "/path/to/project.zip"
}
```

**Returns:**
- Component count
- Files analyzed
- PlantUML architecture diagram
- Structure summary

### 3. `detect_patterns`
Advanced pattern detection with structure context.

**Usage:**
```json
{
  "code": "python code",
  "structure": "json structure (optional)",
  "enrich": true,
  "provider": "openai"
}
```

**Returns:**
- Pattern detections
- Recommendations with UML
- Before/after diagrams

### 4. `get_refactoring_proposal`
Get AI-powered refactoring recommendations.

**Usage:**
```json
{
  "code": "python code",
  "instruction": "Extract Strategy pattern for payments",
  "provider": "sambanova"
}
```

**Returns:**
- Detailed proposal
- Refactoring suggestions
- Code improvements

### 5. `generate_uml`
Generate PlantUML diagrams for code structure.

**Usage:**
```json
{
  "code": "python code",
  "include_methods": true
}
```

**Returns:**
- PlantUML diagram code
- Component count
- Diagram type

## 🔑 Environment Variables

Required for deployment:
```bash
NEBIUS_API_KEY=your_key           # Primary LLM
OPENAI_API_KEY=your_key          # Optional
SAMBANOVA_API_KEY=your_key       # Optional
BLAXCEL_AUTH_TOKEN=your_token    # Blaxcel auth
```

## 📊 Supported Patterns

- **Singleton**: Single instance pattern
- **Factory**: Object creation pattern
- **Strategy**: Behavior selection pattern
- **Observer**: Event notification pattern
- **Builder**: Complex object construction
- **Adapter**: Interface adaptation pattern

## 🤖 LLM Providers

1. **Nebius** (Recommended - Free tier available)
2. **OpenAI** (GPT-4 support)
3. **SambaNova** (Fast inference)

## ✅ Pre-Deployment Checklist

- [ ] Test all tools with `test_mcp.py`
- [ ] Update `mcp.json` with your details
- [ ] Set environment variables
- [ ] Verify GitHub/PyPI links
- [ ] Check `mcp_server.py` syntax
- [ ] Test with sample code
- [ ] Review error handling
- [ ] Document any custom tools

## 🐛 Troubleshooting

### "ModuleNotFoundError: mcp"
```bash
pip install mcp --upgrade
```

### "API Key Invalid"
```bash
# Verify environment variable
echo $NEBIUS_API_KEY

# Re-login to Blaxcel
blaxcel login --token YOUR_TOKEN --force
```

### "Tool not found in server"
```bash
# Check mcp.json validity
python -m json.tool mcp.json

# Verify imports in mcp_server.py
python -c "from mcp_server import *"
```

## 📚 Documentation

- **MCP Spec**: https://spec.modelcontextprotocol.io/
- **Blaxcel Docs**: https://docs.blaxcel.ai/
- **ArchitectAI**: https://github.com/yourusername/architectai-mcp

## 🎯 Next Steps

1. ✅ Test locally
2. ✅ Update configuration files
3. ✅ Set environment variables
4. ✅ Deploy to Blaxcel
5. ✅ Monitor usage
6. ✅ Iterate based on feedback

## 📞 Support

- Issues: https://github.com/yourusername/architectai-mcp/issues
- Docs: https://github.com/yourusername/architectai-mcp/wiki
- Email: team@architectai.dev

---

**Ready to deploy?** Run `blaxcel push --config mcp.json` 🚀
