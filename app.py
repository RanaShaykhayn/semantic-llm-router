import streamlit as st
from src.router_architecture import SemanticLLMRouter
from src.database import get_estimated_cost

@st.cache_resource
def get_router():
    return SemanticLLMRouter()

router = get_router()
st.title("🤖 Semantic LLM Router")

user_input = st.text_input("Ask a question...")

if user_input:
    predicted_route_preview = router.predict_route(user_input)
    # Retrieve cost estimate from database based on the selected route
    est_cost, sample_size = get_estimated_cost(predicted_route_preview)
    if est_cost is not None:
        st.info(f"💡 التكلفة المتوقعة: ${est_cost:.6f} (بناءً على {sample_size} استعلام سابق)")
    else:
        st.info(f"💡 لا توجد بيانات كافية بعد للتقدير ({sample_size} عينة فقط)")

if st.button("إرسال") and user_input:
    result = router.execute_routing(user_input)
    st.write(f"**المسار المستخدم:** {result['route']}")
    st.write(f"**الرد:** {result['response']}")
    st.write(f"**التوفير:** ${result['money_saved']:.5f}")