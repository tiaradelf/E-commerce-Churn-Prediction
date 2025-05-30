import streamlit as st
import pandas as pd
import pickle
import os

def tampilkan_prediction():
    st.subheader("🔍 Prediksi Customer Churn")

    # Load model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            num_address = st.number_input("Number of Address", min_value=1, step=1)
            satisfaction = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])
            day_since_last = st.number_input("Day Since Last Order", min_value=0.0)
            hour_spent = st.number_input("Hour Spend On App", min_value=0.0)
            complain = st.selectbox("Complain", [0, 1])
            login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
            warehouse_to_home = st.number_input("Warehouse To Home", min_value=0.0)
            coupon_used = st.number_input("Coupon Used", min_value=0.0)
            number_of_device = st.number_input("Number of Device Registered", min_value=0, step=1)
            tenure = st.number_input("Tenure (dalam bulan)", min_value=0)
            cashback = st.number_input("Cashback Amount", min_value=0.0)

        with col2:
            order_cat = st.selectbox("Prefered Order Category", ["Laptop & Accessory", "Mobile", "Fashion", "Others"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            gender = st.selectbox("Gender", ["Male", "Female"])
            payment_mode = st.selectbox("Preferred Payment Mode", ["CC", "UPI", "Debit Card", "COD", "Net Banking"])
            order_count = st.number_input("Order Count", min_value=0.0)
            hike_order = st.number_input("Order Amount Hike From Last Year", min_value=0.0)

        # ✅ Tombol submit di dalam form
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_dict = {
            'NumberOfAddress': num_address,
            'SatisfactionScore': satisfaction,
            'DaySinceLastOrder': day_since_last,
            'HourSpendOnApp': hour_spent,
            'Complain': complain,
            'PreferredLoginDevice': login_device,
            'WarehouseToHome': warehouse_to_home,
            'CouponUsed': coupon_used,
            'PreferedOrderCat': order_cat,
            'MaritalStatus': marital_status,
            'CityTier': city_tier,
            'Gender': gender,
            'PreferredPaymentMode': payment_mode,
            'OrderCount': order_count,
            'OrderAmountHikeFromlastYear': hike_order,
            'NumberOfDeviceRegistered': number_of_device,
            'Tenure': tenure,
            'CashbackAmount': cashback
        }

        input_df = pd.DataFrame([input_dict])

        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            if pred == 1:
                st.error(f"Pelanggan kemungkinan **akan churn**. Probabilitas: {prob:.2%}")
            else:
                st.success(f"Pelanggan kemungkinan **tidak churn**. Probabilitas: {prob:.2%}")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")