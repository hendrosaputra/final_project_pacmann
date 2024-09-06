Final Project - Advanced Manchine Learning CLass

Pendahuluan
Algoritma Decision Tree adalah salah satu algoritma yang paling banyak digunakan dalam pembelajaran mesin, terutama untuk tugas klasifikasi dan regresi. Pada esai ini, kita akan membahas penggunaan algoritma ini untuk memprediksi kepuasan pelanggan maskapai dengan menggunakan dataset Air Plane Passenger Data. Dataset ini terdiri dari 129.880 baris data yang mencakup berbagai informasi terkait penerbangan pelanggan, seperti Gender, Customer Type, Age, Class, Flight Distance, dan penilaian terhadap layanan selama penerbangan. Kami akan merangkum komponen utama dari algoritma, pseudocode yang relevan, dan eksperimen dengan hyperparameter untuk meningkatkan performa model.

1. Komponen Pembelajaran dari Algoritma Decision Tree
Untuk membangun model Decision Tree, kita perlu memahami beberapa komponen utama yang berperan penting dalam proses pembelajaran.

a. Impurity (Cost Function)
Impurity adalah ukuran ketidakmurnian atau ketidakpastian di setiap node dalam Decision Tree. Tujuan utama algoritma adalah untuk meminimalkan impurity di setiap node, sehingga dapat memaksimalkan akurasi prediksi. Pada dataset Air Plane Passenger, kita mencoba memprediksi kepuasan pelanggan dengan dua kelas target: satisfied dan neutral or dissatisfied. Untuk kasus klasifikasi biner ini, kriteria impurity yang bisa digunakan adalah:

Gini Impurity: Mengukur probabilitas bahwa sampel yang dipilih secara acak dari sebuah node salah diklasifikasikan. Gini Impurity akan memiliki nilai 0 jika semua sampel dalam node berasal dari kelas yang sama.
Entropy: Ukuran lain dari impurity yang didasarkan pada teori informasi. Semakin rendah nilai entropy, semakin murni node tersebut.
b. Split Criteria
Pada setiap node, Decision Tree melakukan pemecahan data (split) berdasarkan nilai fitur tertentu. Pemecahan ini dilakukan untuk mengurangi impurity di setiap cabang yang dihasilkan. Pada dataset ini, fitur seperti Age, Flight Distance, dan Class memainkan peran penting dalam menentukan split terbaik untuk memprediksi kepuasan pelanggan. Setiap split dilakukan dengan memeriksa fitur dan threshold yang menghasilkan penurunan impurity terbesar.

Sebagai contoh, pelanggan dengan kelas penerbangan First Class mungkin lebih mungkin puas dibandingkan dengan pelanggan dari Economy Class. Demikian juga, pelanggan yang mengalami Arrival Delay mungkin lebih cenderung tidak puas. Dengan memecah data berdasarkan fitur-fitur ini, algoritma dapat belajar untuk memprediksi kelas target.

c. Tree Depth
Kedalaman pohon keputusan, atau tree depth, adalah salah satu hyperparameter penting dalam Decision Tree. Semakin dalam pohon, semakin spesifik model dalam menangani data. Namun, pohon yang terlalu dalam dapat menyebabkan overfitting, yaitu ketika model terlalu cocok dengan data latih sehingga kurang mampu generalisasi pada data baru. Sebaliknya, pohon yang terlalu dangkal bisa mengalami underfitting, di mana model tidak cukup fleksibel untuk menangkap pola yang relevan.

Dalam eksperimen dengan dataset ini, kami mencoba berbagai nilai kedalaman untuk menemukan keseimbangan antara underfitting dan overfitting. Jika pohon terlalu dalam, model akan mempelajari detail yang mungkin tidak relevan, seperti outliers atau noise dalam data.

d. Min Samples Split dan Min Samples Leaf
Untuk mencegah Decision Tree menjadi terlalu spesifik pada data, kita dapat mengatur jumlah minimum sampel yang diperlukan untuk memisahkan node (min_samples_split) dan jumlah minimum sampel di leaf node (min_samples_leaf). Dalam dataset pelanggan maskapai, mengatur nilai-nilai ini dapat membantu menjaga agar model tidak memisahkan node berdasarkan sampel yang terlalu kecil, yang bisa membuat prediksi menjadi tidak akurat.

Sebagai contoh, jika hanya ada beberapa sampel pelanggan yang terbang di kelas First Class, maka leaf node dengan sampel yang terlalu sedikit mungkin tidak memberikan informasi yang berguna. Dengan mengatur jumlah minimum sampel, kita dapat menjaga agar setiap node memecahkan data dengan jumlah yang cukup signifikan.

e. Pruning
Pruning adalah teknik yang digunakan untuk mengurangi ukuran pohon setelah pohon dibangun sepenuhnya. Proses ini membantu mengurangi overfitting dengan menghapus node-node yang tidak memberikan kontribusi signifikan terhadap pengurangan impurity. Setelah pohon dibangun berdasarkan dataset ini, pruning dilakukan dengan membandingkan nilai impurity yang dihasilkan oleh node dengan ambang batas tertentu. Node yang tidak memberikan kontribusi yang cukup besar terhadap performa model akan dihapus.

2. Pseudocode untuk Membangun Algoritma Decision Tree
Untuk memahami logika di balik pembelajaran algoritma Decision Tree, kita dapat merujuk pada pseudocode yang merangkum langkah-langkah utama dalam proses fitting dan prediksi.

a. Fitting the Decision Tree
Proses fitting atau pelatihan model Decision Tree dilakukan dengan membagi dataset ke dalam cabang-cabang berdasarkan kriteria impurity yang dipilih. Berikut adalah pseudocode untuk proses fitting:

Input:

Fitur-fitur pelanggan maskapai (seperti Gender, Age, Class, dll.)
Label target (satisfaction)
Hyperparameter seperti kedalaman pohon, jumlah minimum sampel untuk split, dan impurity criterion.
Process:

Mulai dari node root, hitung impurity dari seluruh dataset.
Cari split terbaik berdasarkan fitur dan threshold yang menghasilkan penurunan impurity terbesar.
Bagi dataset menjadi dua bagian: data yang memenuhi threshold dan yang tidak.
Ulangi proses ini secara rekursif untuk setiap cabang hingga mencapai kedalaman maksimum atau tidak ada lagi split yang signifikan.
Setelah pohon dibangun, panggil fungsi pruning untuk menghilangkan node yang tidak signifikan.
Output:

Pohon keputusan terlatih yang dapat digunakan untuk prediksi.
b. Predicting with the Decision Tree
Setelah pohon keputusan terlatih, prediksi dapat dilakukan dengan menavigasikan data baru melalui pohon. Berikut adalah pseudocode untuk proses prediksi:

Input:

Data baru yang berisi fitur pelanggan maskapai.
Pohon keputusan yang sudah terlatih.
Process:

Untuk setiap sampel baru, mulai dari root node dan periksa apakah node tersebut merupakan leaf.
Jika node bukan leaf, periksa fitur yang digunakan untuk split di node tersebut.
Berdasarkan nilai fitur sampel, tentukan apakah data harus dipindahkan ke cabang kiri atau kanan.
Ulangi proses ini hingga mencapai leaf node, di mana prediksi dibuat berdasarkan kelas mayoritas di node tersebut.
Output:

Prediksi untuk setiap sampel, misalnya apakah pelanggan akan satisfied atau neutral or dissatisfied.
3. Eksperimen Hyperparameter dan Peningkatan Performa Model
Setelah membangun model Decision Tree, eksperimen dengan hyperparameter dilakukan untuk meningkatkan performa prediksi kepuasan pelanggan. Beberapa hyperparameter yang diuji adalah sebagai berikut:

a. Criterion
Pada dataset ini, kita dapat mencoba menggunakan berbagai kriteria impurity, seperti Gini dan Entropy. Setiap kriteria memberikan pendekatan berbeda dalam menghitung impurity, dan kinerjanya dapat berbeda tergantung pada distribusi data.

b. Max Depth
Eksperimen dengan max_depth dilakukan untuk menemukan kedalaman optimal yang tidak terlalu besar agar model tidak overfit, tetapi juga tidak terlalu kecil agar model tidak underfit. Dengan mengatur kedalaman maksimum, kita dapat mencegah model mempelajari pola yang terlalu spesifik pada data latih.

c. Min Samples Split dan Min Samples Leaf
Hyperparameter ini membantu mengontrol jumlah minimum sampel yang diperlukan untuk membagi node atau membuat leaf. Dengan mengatur nilai-nilai ini, kita dapat memastikan bahwa model hanya memecah node jika ada cukup sampel yang signifikan di setiap cabang.

d. Pruning
Proses pruning dilakukan untuk menyederhanakan model setelah pohon dibangun. Ini mencegah pohon menjadi terlalu kompleks, yang dapat menyebabkan overfitting pada data latih. Pruning dilakukan dengan menghapus node yang tidak memberikan kontribusi signifikan terhadap penurunan impurity.

Kesimpulan
Algoritma Decision Tree adalah alat yang kuat untuk memprediksi kepuasan pelanggan dalam berbagai skenario, termasuk penerbangan. Dengan menggunakan dataset Air Plane Passenger Data, kami dapat membangun model prediksi yang baik dengan mengoptimalkan komponen pembelajaran seperti impurity, split criteria, dan depth. Eksperimen dengan hyperparameter seperti criterion, max_depth, dan min_samples_split memungkinkan kita untuk meningkatkan performa model dan menemukan keseimbangan yang tepat antara overfitting dan underfitting. Akhirnya, dengan menggunakan teknik seperti pruning, model dapat dibuat lebih sederhana dan lebih mampu melakukan generalisasi pada data baru.
