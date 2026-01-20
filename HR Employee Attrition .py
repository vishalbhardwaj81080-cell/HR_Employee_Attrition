{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "d9020846-f927-4201-881b-dc1916bfb068",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd            \n",
    "import numpy as np             \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.metrics import accuracy_score,precision_score,classification_report, confusion_matrix\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "78cc89fe-96da-4041-afd0-84b327887da4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Attrition</th>\n",
       "      <th>BusinessTravel</th>\n",
       "      <th>DailyRate</th>\n",
       "      <th>Department</th>\n",
       "      <th>DistanceFromHome</th>\n",
       "      <th>Education</th>\n",
       "      <th>EducationField</th>\n",
       "      <th>EmployeeCount</th>\n",
       "      <th>EmployeeNumber</th>\n",
       "      <th>...</th>\n",
       "      <th>RelationshipSatisfaction</th>\n",
       "      <th>StandardHours</th>\n",
       "      <th>StockOptionLevel</th>\n",
       "      <th>TotalWorkingYears</th>\n",
       "      <th>TrainingTimesLastYear</th>\n",
       "      <th>WorkLifeBalance</th>\n",
       "      <th>YearsAtCompany</th>\n",
       "      <th>YearsInCurrentRole</th>\n",
       "      <th>YearsSinceLastPromotion</th>\n",
       "      <th>YearsWithCurrManager</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>41</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>1102</td>\n",
       "      <td>Sales</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>80</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Frequently</td>\n",
       "      <td>279</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>80</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>10</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>1373</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>Other</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>...</td>\n",
       "      <td>2</td>\n",
       "      <td>80</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Frequently</td>\n",
       "      <td>1392</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>80</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>27</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>591</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>Medical</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>80</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 35 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age Attrition     BusinessTravel  DailyRate              Department  \\\n",
       "0   41       Yes      Travel_Rarely       1102                   Sales   \n",
       "1   49        No  Travel_Frequently        279  Research & Development   \n",
       "2   37       Yes      Travel_Rarely       1373  Research & Development   \n",
       "3   33        No  Travel_Frequently       1392  Research & Development   \n",
       "4   27        No      Travel_Rarely        591  Research & Development   \n",
       "\n",
       "   DistanceFromHome  Education EducationField  EmployeeCount  EmployeeNumber  \\\n",
       "0                 1          2  Life Sciences              1               1   \n",
       "1                 8          1  Life Sciences              1               2   \n",
       "2                 2          2          Other              1               4   \n",
       "3                 3          4  Life Sciences              1               5   \n",
       "4                 2          1        Medical              1               7   \n",
       "\n",
       "   ...  RelationshipSatisfaction StandardHours  StockOptionLevel  \\\n",
       "0  ...                         1            80                 0   \n",
       "1  ...                         4            80                 1   \n",
       "2  ...                         2            80                 0   \n",
       "3  ...                         3            80                 0   \n",
       "4  ...                         4            80                 1   \n",
       "\n",
       "   TotalWorkingYears  TrainingTimesLastYear WorkLifeBalance  YearsAtCompany  \\\n",
       "0                  8                      0               1               6   \n",
       "1                 10                      3               3              10   \n",
       "2                  7                      3               3               0   \n",
       "3                  8                      3               3               8   \n",
       "4                  6                      3               3               2   \n",
       "\n",
       "  YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager  \n",
       "0                  4                        0                     5  \n",
       "1                  7                        1                     7  \n",
       "2                  0                        0                     0  \n",
       "3                  7                        3                     0  \n",
       "4                  2                        2                     2  \n",
       "\n",
       "[5 rows x 35 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df= pd.read_csv(r\"C:\\Users\\Vishal Kumar\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40a26d34-152b-40c6-8fe5-dee9d363c1bc",
   "metadata": {},
   "source": [
    " # Drop unwanted columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f463833-bee8-4dd6-b64e-7f85674f5b99",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.shape\n",
    "df.info()\n",
    "df.isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e4ca5956-e027-45bc-b687-991558794e24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Attrition</th>\n",
       "      <th>BusinessTravel</th>\n",
       "      <th>DailyRate</th>\n",
       "      <th>Department</th>\n",
       "      <th>DistanceFromHome</th>\n",
       "      <th>Education</th>\n",
       "      <th>EducationField</th>\n",
       "      <th>EnvironmentSatisfaction</th>\n",
       "      <th>Gender</th>\n",
       "      <th>...</th>\n",
       "      <th>PerformanceRating</th>\n",
       "      <th>RelationshipSatisfaction</th>\n",
       "      <th>StockOptionLevel</th>\n",
       "      <th>TotalWorkingYears</th>\n",
       "      <th>TrainingTimesLastYear</th>\n",
       "      <th>WorkLifeBalance</th>\n",
       "      <th>YearsAtCompany</th>\n",
       "      <th>YearsInCurrentRole</th>\n",
       "      <th>YearsSinceLastPromotion</th>\n",
       "      <th>YearsWithCurrManager</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>41</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>1102</td>\n",
       "      <td>Sales</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>2</td>\n",
       "      <td>Female</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Frequently</td>\n",
       "      <td>279</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>3</td>\n",
       "      <td>Male</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>10</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>1373</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>Other</td>\n",
       "      <td>4</td>\n",
       "      <td>Male</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Frequently</td>\n",
       "      <td>1392</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>Life Sciences</td>\n",
       "      <td>4</td>\n",
       "      <td>Female</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>7</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>27</td>\n",
       "      <td>No</td>\n",
       "      <td>Travel_Rarely</td>\n",
       "      <td>591</td>\n",
       "      <td>Research &amp; Development</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>Medical</td>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age Attrition     BusinessTravel  DailyRate              Department  \\\n",
       "0   41       Yes      Travel_Rarely       1102                   Sales   \n",
       "1   49        No  Travel_Frequently        279  Research & Development   \n",
       "2   37       Yes      Travel_Rarely       1373  Research & Development   \n",
       "3   33        No  Travel_Frequently       1392  Research & Development   \n",
       "4   27        No      Travel_Rarely        591  Research & Development   \n",
       "\n",
       "   DistanceFromHome  Education EducationField  EnvironmentSatisfaction  \\\n",
       "0                 1          2  Life Sciences                        2   \n",
       "1                 8          1  Life Sciences                        3   \n",
       "2                 2          2          Other                        4   \n",
       "3                 3          4  Life Sciences                        4   \n",
       "4                 2          1        Medical                        1   \n",
       "\n",
       "   Gender  ...  PerformanceRating  RelationshipSatisfaction  StockOptionLevel  \\\n",
       "0  Female  ...                  3                         1                 0   \n",
       "1    Male  ...                  4                         4                 1   \n",
       "2    Male  ...                  3                         2                 0   \n",
       "3  Female  ...                  3                         3                 0   \n",
       "4    Male  ...                  3                         4                 1   \n",
       "\n",
       "  TotalWorkingYears  TrainingTimesLastYear WorkLifeBalance  YearsAtCompany  \\\n",
       "0                 8                      0               1               6   \n",
       "1                10                      3               3              10   \n",
       "2                 7                      3               3               0   \n",
       "3                 8                      3               3               8   \n",
       "4                 6                      3               3               2   \n",
       "\n",
       "   YearsInCurrentRole  YearsSinceLastPromotion YearsWithCurrManager  \n",
       "0                   4                        0                    5  \n",
       "1                   7                        1                    7  \n",
       "2                   0                        0                    0  \n",
       "3                   7                        3                    0  \n",
       "4                   2                        2                    2  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.drop([\"EmployeeCount\",\"EmployeeNumber\",\"Over18\",\"StandardHours\"],axis=1,inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d187352-0e17-4bce-ae81-e1309e67c7c1",
   "metadata": {},
   "source": [
    "# Encode Target Columnn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "97732825-4428-458f-9bf4-964ff8df51d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Attrition\"]=df[\"Attrition\"].map({\"Yes\":1,\"No\":0})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5576d26c-65b5-4976-990a-e1d7f33a7c9c",
   "metadata": {},
   "source": [
    "# Encoding Categorial Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "79779041-fd3b-4c70-b5f1-93abfd36caae",
   "metadata": {},
   "outputs": [],
   "source": [
    "le = LabelEncoder()\n",
    "for col in df.select_dtypes(include=\"object\").columns:\n",
    "    df[col]=le.fit_transform(df[col])\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2bb3785b-8d40-4fe6-831a-290bc61b18ca",
   "metadata": {},
   "source": [
    "# feature vs target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9f1f692d-f772-4de8-83c5-6146295f0c54",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop(\"Attrition\",axis=1)\n",
    "y = df[\"Attrition\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d45d5416-c92c-49b7-a83e-f06cf8ef38ef",
   "metadata": {},
   "source": [
    "# Train-Test Split "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "21f1fd8d-f702-4866-a056-334c35112a76",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify=y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd221a82-4f8b-4070-9f29-ef275a4bb4c2",
   "metadata": {},
   "source": [
    "# feature scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "001a83bd-e3a4-4330-9869-1f518fbf0643",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "x_train = scaler.fit_transform(x_train)\n",
    "x_test = scaler.transform(x_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9a060cc-7691-441f-b4d5-36cd0b3fb5ae",
   "metadata": {},
   "source": [
    "# Model ( 1- Regression)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "b153259d-1e37-41c6-8f23-73add859f21c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8741496598639455\n",
      "Precision 0.6923076923076923\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.89      0.97      0.93       247\n",
      "           1       0.69      0.38      0.49        47\n",
      "\n",
      "    accuracy                           0.87       294\n",
      "   macro avg       0.79      0.68      0.71       294\n",
      "weighted avg       0.86      0.87      0.86       294\n",
      "\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression()\n",
    "lr.fit(x_train, y_train)\n",
    "\n",
    "y_pred_lr = lr.predict(x_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred_lr))\n",
    "print(\"Precision\", precision_score(y_test,y_pred_lr))\n",
    "print(classification_report(y_test, y_pred_lr))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3a80ef0-d546-455c-b343-ec5b972e346b",
   "metadata": {},
   "source": [
    "# Model(2-Random Forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "20664b5b-6d87-4584-8004-7349b322453b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8367346938775511\n",
      "Precision 0.46153846153846156\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.85      0.97      0.91       247\n",
      "           1       0.46      0.13      0.20        47\n",
      "\n",
      "    accuracy                           0.84       294\n",
      "   macro avg       0.66      0.55      0.55       294\n",
      "weighted avg       0.79      0.84      0.80       294\n",
      "\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(n_estimators=200, random_state=42)\n",
    "rf.fit(x_train, y_train)\n",
    "\n",
    "y_pred_rf = rf.predict(x_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred_rf))\n",
    "print(\"Precision\", precision_score(y_test,y_pred_rf))\n",
    "print(classification_report(y_test, y_pred_rf))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c10e841-6621-46be-9ec7-0ae5fe0ddf61",
   "metadata": {},
   "source": [
    "# Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "e996fefc-6433-4a51-95e3-3cea158594c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, y_pred_rf)\n",
    "\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca688c1f-cdce-4911-a672-5d2453f284c5",
   "metadata": {},
   "source": [
    "# Importance Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "8345047d-d992-4a4d-bf4b-da2dad0777e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MonthlyIncome         0.077536\n",
       "Age                   0.069066\n",
       "TotalWorkingYears     0.063224\n",
       "DailyRate             0.056067\n",
       "HourlyRate            0.052921\n",
       "MonthlyRate           0.050547\n",
       "DistanceFromHome      0.048588\n",
       "YearsAtCompany        0.042428\n",
       "OverTime              0.040214\n",
       "NumCompaniesWorked    0.037178\n",
       "dtype: float64"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature_importance = pd.Series(\n",
    "    rf.feature_importances_,index=x.columns).sort_values(ascending=False)\n",
    "feature_importance.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26413489-8c91-454c-990d-a2c5a5d827a5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
