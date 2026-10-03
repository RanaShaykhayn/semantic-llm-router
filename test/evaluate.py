import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database import log_evaluation_run
from src.router_architecture import SemanticLLMRouter
from datasets import load_dataset
from langdetect import detect, LangDetectException


def route_to_label(route: str) -> int | None:
    if route in ("Local Simulator", "gemini-2.5-flash", "Semantic Cache (Hit)"):
        return 0
    elif route == "gemini-2.5-pro":
        return 1
    return None


def is_arabic_or_english(text: str) -> bool:
    try:
        return detect(text) in ("ar", "en")
    except LangDetectException:
        return False


def evaluate():
    print("جاري تحميل البيانات...")
    dataset = load_dataset("mrzaizai2k/gpt_routing")
    data = dataset["train"]

    print("جاري الفلترة (عربي/إنجليزي فقط)...")
    filtered_data = [row for row in data if is_arabic_or_english(row['text'])]
    total = len(filtered_data)
    print(f"بعد الفلترة: {total} عينة من أصل {len(data)}")

    router = SemanticLLMRouter()
    correct_predictions = 0
    excluded = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    print(f"بدء التقييم على {total} عينة...")

    for i in range(total):
        row = filtered_data[i]
        user_input = row['text']
        true_label = row['label']

        result_route = router.predict_route(user_input)
        predicted_label = route_to_label(result_route)

        if predicted_label is None:
            excluded += 1
            print(f"عينة {i+1}: مستثناة ({result_route})")
        else:
            is_correct = predicted_label == true_label
            if predicted_label==1 and true_label==1:
                tp+=1
            elif predicted_label==1 and true_label==0:
                fp+=1
            elif predicted_label==0 and true_label==1:
                fn+=1
            elif predicted_label==0 and true_label==0:
                tn+=1
            correct_predictions += is_correct
            print(f"عينة {i+1}: التوقع={predicted_label}, الصحيح={true_label}, {'✅' if is_correct else '❌'}")

        time.sleep(2)

    evaluated = total - excluded
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0
    f1_score=2*(precision*recall)/(precision+recall) if (precision+recall) >0 else 0
    accuracy=(correct_predictions / evaluated * 100) if evaluated > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1_score:.2f}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted: Pro   Predicted: Cheap")
    print(f"Actually: Pro         {tp:<15} {fn}")
    print(f"Actually: Cheap       {fp:<15} {tn}")
    log_evaluation_run({
    "total_samples": total,
    "excluded_samples": excluded,
    "accuracy": accuracy,
    "precision_score": precision,
    "recall": recall,
    "f1": f1_score,
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn,
})
if __name__ == "__main__":
    evaluate()
