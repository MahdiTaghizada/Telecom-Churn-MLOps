# Kod üzrə izah

## Ümumi axın

`data.py` açıq IBM CSV faylını yükləyir, sütunları, target-i və rəqəmsal
dəyişənləri yoxlayır. `splitting.py` stratified bölgü edir ki, train,
validation və test hissələrində churn balansı mümkün qədər qorunsun.

## Leakage necə qarşısı alınır?

`preprocessing.py` rəqəmsal sütunlar üçün median imputer və scaler, kateqorik
sütunlar üçün most-frequent imputer və `OneHotEncoder(handle_unknown="ignore")`
qurur. Bu addımlar model ilə birlikdə sklearn `Pipeline` daxilindədir.
Parametrlər yalnız train hissəsində öyrənilir; validation, test və API girişi
eyni artıq öyrənilmiş addımlardan keçir. `customerID` feature siyahısına daxil
edilmir.

## Model seçimi və MLflow

`models.py` Logistic Regression və Random Forest axtarışlarını yaradır. İki
model validation F1 ilə müqayisə olunur; test yalnız seçimdən sonra bir dəfə
hesablanır. `tracking.py` parametrləri, metrikləri, evaluation fayllarını və
modeli MLflow-a yazır, seçilmiş modeli Registry-də qeyd edir və `champion`
alias-ını yeniləyir.

## API

`api.py` startup zamanı `models/champion` bundle-ını bir dəfə yükləyir.
`/predict` input-u Pydantic ilə yoxlayır və prediction, ehtimal və model
versiyasını qaytarır. Model yüklənməzsə health `degraded`, prediction isə
503 olur. `/metrics` Prometheus formatında aşağı kardinallıqlı route/status
label-ları ilə HTTP sayını, xətaları və müddəti, həmçinin prediction və
inference müddətini verir.

## Drift məntiqi

`drift.py` reference kimi modelin deterministic training split-inin yalnız
feature hissəsini istifadə edir. Control current data eyni paylanmadan
sample-dır və drift gözlənilmir. `SYNTHETIC` current data `MonthlyCharges`,
`tenure`, `Contract` və `PaymentMethod` paylanmalarını qəsdən dəyişir.
Rəqəmlilər üçün iki-nümunəli KS, kateqoriklər üçün chi-square testi və
`p < 0.05` həddi seçilib. Data drift input paylanmasının dəyişməsidir;
concept drift isə input-target əlaqəsinin dəyişməsidir. Drift tapılması model
keyfiyyətinin pisləşdiyini təkbaşına sübut etmir; bunun üçün yeni etiketli data
ilə ayrıca evaluation lazımdır.
