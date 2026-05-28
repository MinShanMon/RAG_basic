import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    'C:/Users/Window/OneDrive/Documents/DevOps',
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    show_progress=False,
    silent_errors=True,
)
docs = loader.load()
sources = sorted(set(d.metadata.get("source", "") for d in docs))
print(f"Total chunks: {len(docs)}, unique files: {len(sources)}")
for s in sources:
    if "Knowledge" in s or "FULL" in s or "markdown file" in s.lower():
        print("  FOUND:", s)
