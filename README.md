
Demo code using C API for tensorflow


Install TensorFlow for C
https://www.tensorflow.org/install/lang_c

official build
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-1.12.0.zip
(cudnn mininmum 7.2.1)

all official builds
https://storage.googleapis.com/tensorflow

tensoflow.dll - for x64 only

possible find *tensorflow*.pyd for python and rename it to tensoflow.dll (gpu or cpu any version)

tensorflow.lib need for link applications
creation lib file from dll file
https://adrianhenke.wordpress.com/2008/12/05/create-lib-file-from-dll/

commands for creattion lib file
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\dumpbin" /exports tensorflow.dll > f.txt
(edit content f.txt to tensorflow.def)
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\lib" /MACHINE:x64 /def:tensorflow.def /OUT:tensorflow.lib


samples:
https://stackoverflow.com/questions/44378764/hello-tensorflow-using-the-c-api
https://stackoverflow.com/questions/41688217/how-to-load-a-graph-with-tensorflow-so-and-c-api-h-in-c-language/41688506
https://github.com/Neargye/hello_tf_c_api

