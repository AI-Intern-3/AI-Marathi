**डेटा प्रीप्रोसेसिंग आणि क्लिनिंग प्रक्रिया AI मध्ये - मराठीत**

डेटा प्रीप्रोसेसिंग (Data Preprocessing) आणि क्लिनिंग (Cleaning) ही AI आणि Machine Learning प्रोजेक्ट्समध्ये अत्यंत महत्त्वाची पायरी आहे. या प्रक्रियेमध्ये, कच्च्या (raw) डेटा सेटला योग्य स्वरूपात आणण्यासाठी विविध तंत्रांचा वापर केला जातो, ज्यामुळे मॉडेलची कार्यक्षमता सुधारते.

### डेटा प्रीप्रोसेसिंग म्हणजे काय?

डेटा प्रीप्रोसेसिंग ही प्रक्रिया कच्च्या डेटा सेटवर विविध तंत्रांचा वापर करून त्याचे रूपांतर स्वच्छ आणि योग्य डेटा सेटमध्ये करण्यासाठी केली जाते. यात डेटा क्लिनिंग, डेटा ट्रान्सफॉर्मेशन, फीचर स्केलिंग, नॉर्मलायझेशन इत्यादी तंत्रांचा समावेश आहे.

### डेटा प्रीप्रोसेसिंगचे महत्त्व

1. **मिसिंग व्हॅल्यूज (Missing Values):** कच्च्या डेटामध्ये काही व्हॅल्यूज मिसिंग असू शकतात, ज्यामुळे मॉडेलची कार्यक्षमता कमी होते.
2. **नॉइज (Noise):** अनावश्यक डेटा पॉइंट्स किंवा एरर जे मॉडेलच्या ट्रेनिंगमध्ये अडथळा आणू शकतात.
3. **स्केलिंग (Scaling):** विविध फीचर्सच्या व्हॅल्यूजमध्ये असमानता असते, ज्यामुळे मॉडेलच्या कॉन्झर्जन्समध्ये समस्या येऊ शकते.
4. **कैटेगोरिकल डेटा (Categorical Data):** टेक्स्ट फॉर्ममध्ये असलेला डेटा ज्याला न्यूमेरिकल व्हॅल्यूमध्ये रूपांतर करणे आवश्यक असते.
5. **नॉर्मलायझेशन (Normalization):** डेटा एका विशिष्ट रेंजमध्ये आणणे.

### डेटा प्रीप्रोसेसिंगची प्रक्रिया

#### 1. **डेटा क्लिनिंग (Data Cleaning)**

डेटा क्लिनिंग ही प्रक्रिया डेटामधील एरर, मिसिंग व्हॅल्यूज आणि नॉइज काढून टाकण्याची प्रक्रिया आहे.

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# उदाहरण डेटा सेट लोड करणे
data = {'Name': ['Amit', 'Neha', 'Raj', 'Priya', 'Nikhil'],
        'Age': [25, 30, None, 22, 35],
        'Salary': [50000, 60000, 55000, None, 52000]}

df = pd.DataFrame(data)

# मिसिंग व्हॅल्यूज काढणे
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)
```

#### 2. **डेटा ट्रान्सफॉर्मेशन (Data Transformation)**

डेटा ट्रान्सफॉर्मेशनमध्ये डेटा सेटला मॉडेलसाठी योग्य स्वरूपात बदलणे समाविष्ट आहे.

##### (a) **कैटेगोरिकल डेटा एनकोडिंग (Categorical Data Encoding)**

```python
from sklearn.preprocessing import LabelEncoder

# उदाहरण डेटा सेट
data = {'City': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Mumbai']}
df = pd.DataFrame(data)

# Label Encoding
label_encoder = LabelEncoder()
df['City_encoded'] = label_encoder.fit_transform(df['City'])

print(df)
```

##### (b) **फीचर स्केलिंग (Feature Scaling)**

```python
from sklearn.preprocessing import StandardScaler

# उदाहरण डेटा सेट
data = {'Age': [25, 30, 22, 35, 29],
        'Salary': [50000, 60000, 55000, 52000, 58000]}

df = pd.DataFrame(data)

# Standardization
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print(df)
```

##### (c) **नॉर्मलायझेशन (Normalization)**

नॉर्मलायझेशन म्हणजे डेटा एका विशिष्ट रेंजमध्ये (उदा. 0 ते 1) आणणे. MinMaxScaler वापरून नॉर्मलायझेशन करता येते.

```python
from sklearn.preprocessing import MinMaxScaler

# उदाहरण डेटा सेट
data = {'Age': [25, 30, 22, 35, 29],
        'Salary': [50000, 60000, 55000, 52000, 58000]}

df = pd.DataFrame(data)

# MinMaxScaler वापरून नॉर्मलायझेशन
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print(df)
```

#### 3. **आउटलायर्स काढणे (Removing Outliers)**

आउटलायर्स म्हणजे डेटामधील अनावश्यक आणि अत्यधिक वेगळे डेटा पॉइंट्स जे मॉडेलच्या ट्रेनिंगला बाधा आणू शकतात. Z-score किंवा IQR (Interquartile Range) वापरून आउटल्यर्स काढता येतात.

```python
import numpy as np

# उदाहरण डेटा सेट
data = {'Age': [25, 30, 22, 35, 29, 100],  # 100 हा आउटलायर आहे
        'Salary': [50000, 60000, 55000, 52000, 58000, 49000]}

df = pd.DataFrame(data)

# Z-score वापरून आउटल्यर्स काढणे
from scipy import stats
z_scores = np.abs(stats.zscore(df))
df = df[(z_scores < 3).all(axis=1)]

print(df)
```

#### 4. **डेटा विभाजन (Data Splitting)**

डेटा ट्रेनिंग आणि टेस्टिंग सेटमध्ये विभाजित करणे महत्वाचे आहे जेणेकरून मॉडेलची जनरलायझेशन क्षमता तपासता येईल.

```python
from sklearn.model_selection import train_test_split

# उदाहरण डेटा सेट
data = {'Age': [25, 30, 22, 35, 29],
        'Salary': [50000, 60000, 55000, 52000, 58000],
        'Purchased': [0, 1, 0, 1, 1]}

df = pd.DataFrame(data)

# फीचर्स आणि लेबल्स विभाजित करणे
X = df[['Age', 'Salary']]
y = df['Purchased']

# ट्रेनिंग आणि टेस्टिंग सेटमध्ये डेटा विभाजित करणे
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Train Data:\n", X_train)
print("Test Data:\n", X_test)
```

### सारांश

डेटा प्रीप्रोसेसिंग आणि क्लिनिंग प्रक्रिया AI मध्ये अत्यंत महत्त्वाची आहे कारण ही प्रक्रिया डेटा सेटला स्वच्छ, संरचित आणि मॉडेलसाठी योग्य स्वरूपात बनवते. योग्य प्रकारे डेटा प्रीप्रोसेसिंग केल्याने मॉडेलची कार्यक्षमता आणि अचूकता सुधारते. यामध्ये मिसिंग व्हॅल्यूज काढणे, डेटा ट्रान्सफॉर्मेशन, नॉर्मलायझेशन, आउटल्यर्स काढणे आणि डेटा विभाजन यांचा समावेश आहे.
