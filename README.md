# SALES ANALYTICS DASHBOARD

Interactive dashboard untuk analisis performa sales dan prediksi probabilitas deal berdasarkan data kunjungan sales.

## 🚀 Features

### 📊 **Interactive Dashboard**
- **Real-time filtering** berdasarkan tanggal, segmen, level sales, progress stage, customer status, dan sales person
- **Key Performance Metrics** yang update otomatis
- **Interactive Charts** dengan Plotly untuk visualisasi yang menarik
- **Responsive Design** yang mobile-friendly

### 📈 **Analytics Modules**
1. **Segment Analysis** - Performa per segmen customer
2. **Sales Performance** - Ranking dan analisis individual sales
3. **Progress Analysis** - Conversion rate per tahap sales
4. **Predictive Analytics** - Machine learning untuk prediksi deal probability

### 🔮 **Predictive Model**
- **Machine Learning Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Feature Engineering**: Interaction features, scaled targets, encoded categories
- **Real-time Prediction**: Input parameter dan dapatkan probabilitas deal
- **Model Performance Metrics** dan Feature Importance

## 📁 File Structure

```
📁 Sales Analytics Dashboard/
├── 📄 dashboard_main.py          # Main dashboard application
├── 📄 dashboard_utils.py         # Utility classes untuk data processing & charts
├── 📄 predictive_model.py        # Machine learning model untuk prediksi
├── 📄 requirements.txt           # Python packages yang dibutuhkan
├── 📄 test_dashboard.py          # Test script untuk verifikasi dashboard
├── 📄 README.md                  # Documentation (file ini)
└── 📄 sales_visits_finalbgt_enriched.csv  # Data sales (harus ada)
```

## 🛠️ Installation & Setup

### 1. **Install Python Packages**
```bash
pip install -r requirements.txt
```

### 2. **Pastikan Data File Tersedia**
Pastikan file `sales_visits_finalbgt_enriched.csv` ada di folder yang sama dengan script.

### 3. **Test Dashboard (Optional)**
```bash
python test_dashboard.py
```

### 4. **Run Dashboard**
```bash
streamlit run dashboard_main.py
```

### 4. **Akses Dashboard**
Buka browser dan kunjungi: `http://localhost:8501`

## 🎯 How to Use

### **1. Filters & Controls**
- **Date Range**: Pilih rentang tanggal untuk analisis
- **Customer Segment**: Filter berdasarkan segmen (Government, Telco, Private, SOE, Regional)
- **Sales Level**: Filter AM atau EAM
- **Progress Stage**: Filter tahapan sales (Inisiasi → Paska Deal)
- **Customer Status**: Filter status customer (Baru, Lama, Win-Back)
- **Sales Person**: Filter sales person spesifik

### **2. Key Metrics Dashboard**
Dashboard menampilkan 5 metric utama:
- 📊 **Total Visits**: Total kunjungan sales
- 🎯 **Total Deals**: Total deal yang berhasil
### **2. Key Metrics Dashboard**
- 🔢 **Total Visits**: Jumlah total kunjungan sales
- ✅ **Won Deals**: Jumlah deal yang berhasil (Status: "Deal")
- ⏳ **Ongoing Deals**: Jumlah deal yang sedang berjalan (Status: "Berpotensi Deal")
- 📈 **Win Rate**: Persentase keberhasilan deal
- ⏱️ **Avg Visits to Close**: Rata-rata kunjungan untuk closing
- 💰 **Won Value**: Total nilai deal yang berhasil dalam juta rupiah
- 🎯 **Potential Value**: Total nilai potensial dari ongoing deals

### **3. Performance Charts**
- **Deal Status Distribution**: Breakdown status deal (Won/Ongoing/Lost)
- **Win Rate by Segment**: Performa per segmen customer
- **Sales Funnel**: Analisis funnel dari Inisiasi → Deal
- **Monthly Trend**: Trend performa bulanan
- **Sales Performance Scatter**: Visualisasi visits vs deals per sales
- **Deal Probability Heatmap**: Probabilitas deal per progress vs visit number

### **4. Updated Data Structure**
Dashboard sekarang menggunakan data dengan:
- ✨ **Nilai_Kontrak**: Nilai kontrak aktual dalam rupiah
- 🎯 **Status_Kontrak**: "Berpotensi Deal", "Deal", "Cancel"
- 📊 **Enhanced Analytics**: Nilai ditampilkan dalam juta rupiah untuk keterbacaan

### **4. Detailed Analysis Tabs**

#### **📊 Segment Analysis**
- Tabel performa per segmen
- Chart perbandingan segment
- Metrics: Win Rate, Total Visits, Deal Value, dll.

#### **👨‍💼 Sales Performance**
- Ranking sales berdasarkan win rate
- Perbandingan AM vs EAM
- Performance matrix dan scatter plot

#### **🎯 Progress Analysis**
- Conversion rate per tahap progress
- Average visit number per stage
- Funnel analysis mendalam

#### **🔮 Predictive Model**
- **Input Parameters**:
  - Progress Stage
  - Visit Number (1-6)
  - Customer Segment
  - Customer Status
  - Sales Level
  - Target Sales & Segment Value
- **Output**: Probabilitas deal dalam persentase
- **Model Metrics**: Akurasi, CV Score, dll.
- **Feature Importance**: Faktor yang paling berpengaruh

## 📊 Sample Insights

### **Key Findings dari Data**
- **Best Segment**: Telco (47% win rate)
- **Optimal Visit**: Visit ke-5 dan ke-6 memiliki conversion rate tertinggi
- **Best Progress Stage**: Negosiasi dan Paska Deal
- **Top Features**: Target Sales Value, Customer Status, Segment Type

### **Actionable Recommendations**
1. **Focus on Telco segment** untuk conversion rate terbaik
2. **Optimize sales process** untuk mencapai visit ke-5+
3. **Prioritize Negosiasi stage** karena conversion rate tinggi
4. **Train sales** pada Customer Status handling

## 🔧 Technical Details

### **Data Processing**
- Automatic data type conversion
- Date parsing dan formatting
- Missing value handling
- Feature engineering untuk ML

### **Machine Learning Pipeline**
- Train-test split (70-30)
- Cross-validation (5-fold)
- Model comparison dan selection
- Feature importance calculation

### **Chart Generation**
- Plotly untuk interactive charts
- Responsive design
- Color-coded metrics
- Real-time updates

## 🚨 Troubleshooting

### **Common Issues**

1. **File tidak ditemukan**
   ```
   ❌ File 'sales_visits_enriched_csv.csv' not found!
   ```
   **Solution**: Pastikan file CSV ada di folder yang sama dengan script

2. **Package tidak terinstall**
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Run `pip install -r requirements.txt`

3. **Dashboard tidak loading**
   **Solution**: Check terminal untuk error messages, pastikan port 8501 tidak digunakan

4. **Predictive model error**
   **Solution**: Pastikan data memiliki semua kolom yang dibutuhkan dan tidak ada missing values

## 📈 Future Enhancements

### **Planned Features**
- [ ] Export reports ke PDF/Excel
- [ ] Advanced filtering options
- [ ] Real-time data integration
- [ ] Email alerts untuk performance
- [ ] Mobile app version
- [ ] Advanced ML models (XGBoost, Neural Networks)

### **Data Enhancements**
- [ ] Historical trend analysis
- [ ] Competitive analysis
- [ ] Customer satisfaction metrics
- [ ] Revenue forecasting

## 👥 Support

Untuk pertanyaan atau issue:
1. Check troubleshooting section
2. Review error messages di terminal
3. Pastikan semua requirements terpenuhi
4. Verify data format dan structure

## 📝 License

Internal use untuk analisis sales performance dan business intelligence.

---

**📊 Sales Analytics Dashboard** | Built with Streamlit & Python | Data-driven insights untuk optimasi sales performance
