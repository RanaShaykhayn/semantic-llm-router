REFERENCE_GREETINGS = [
    # English
    "hello",
    "hi there",
    "good morning",
    "how are you doing?",
    # Arabic
    "مرحبا",
    "أهلاً",
    "صباح الخير",
    "كيف حالك؟",
]

REFERENCE_COMPLEX = [
    # English
    "explain quantum computing in simple terms",
    "write a python script for scraping",
    "design a system architecture for a web app",
    "analyze the philosophical implications of AI",
    # Arabic
    "اشرح الحوسبة الكمومية ببساطة",
    "اكتب سكربت بايثون لجمع البيانات",
    "صمم بنية نظام لتطبيق ويب",
    "حلل الآثار الفلسفية للذكاء الاصطناعي",
]

CACHE_TTL_SECONDS = 86400
CACHE_MAX_SIZE = 100

SIMILARITY_THRESHOLD_COMPLEX = 0.62
SIMILARITY_THRESHOLD_GREETING = 0.60
SIMILARITY_THRESHOLD_CACHE = 0.78

GEMINI_FLASH_INPUT_PRICE = 0.30
GEMINI_FLASH_OUTPUT_PRICE = 2.50
GEMINI_PRO_INPUT_PRICE = 2.00
GEMINI_PRO_OUTPUT_PRICE = 12.00
MAX_LATENCY_MS = 2000