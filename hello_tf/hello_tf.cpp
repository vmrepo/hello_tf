// hello_tf.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c_api.h>

#include <opencv2/opencv.hpp>

using namespace cv;

TF_Buffer* read_file( const char* file );

void free_buffer( void* data, size_t length ) {
	free( data );
}

int imagenet_classify() {

	TF_Buffer* graph_def = read_file( "classify_image_graph_def.pb" );
	TF_Graph* graph = TF_NewGraph();

	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef( graph, graph_def, opts, status );
	TF_DeleteImportGraphDefOptions( opts );
	if (TF_GetCode( status ) != TF_OK) {
		fprintf( stderr, "ERROR: Unable to import graph %s\n", TF_Message( status ) );
		return 1;
	}
	TF_DeleteBuffer( graph_def );
	fprintf( stdout, "Successfully imported graph\n" );

	TF_Operation * input_op = TF_GraphOperationByName( graph, "DecodeJpeg/contents" );
	struct TF_Output input;
	input.oper = input_op;
	input.index = 0;

	if (input.oper == nullptr) {
		fprintf( stderr, "Can't init input_op" );
		return 1;
	}

	TF_Buffer* input_buf = read_file( "cropped_panda.jpg" );
	TF_Tensor* input_tensor = TF_AllocateTensor( TF_STRING, 0, 0, 8 + TF_StringEncodedSize( input_buf->length ) );
	void* tensor_data = TF_TensorData( input_tensor );
	TF_StringEncode( (char*)input_buf->data, input_buf->length, 8 + (char *)tensor_data, TF_StringEncodedSize( input_buf->length ), status );

	if (TF_GetCode( status ) != TF_OK) {
		printf( "Error: %s\n", TF_Message( status ) );
		return 1;
	}

	memset( tensor_data, 0, 8 );
	TF_DeleteBuffer( input_buf );

	TF_Operation * output_op = TF_GraphOperationByName( graph, "softmax" );
	struct TF_Output output;
	output.oper = output_op;
	output.index = 0;

	if (output.oper == nullptr) {
		fprintf( stderr, "Can't init out_op" );
		return 1;
	}

	TF_Tensor* output_tensor = nullptr;

	TF_SessionOptions * options = TF_NewSessionOptions();
	TF_Session * session = TF_NewSession( graph, options, status );
	TF_DeleteSessionOptions( options );

	TF_SessionRun( session,
		nullptr,
		&input, &input_tensor, 1,
		&output, &output_tensor, 1,
		nullptr, 0,
		nullptr,
		status
	);

	if (TF_GetCode( status ) != TF_OK) {
		printf( "Error: %s\n", TF_Message( status ) );
		return 1;
	}

	TF_CloseSession( session, status );
	TF_DeleteSession( session, status );

	float* out = (float*)TF_TensorData( output_tensor );
	printf( "%e %e %e ... %e %e %e\n", out[0], out[1], out[2], out[1005], out[1006], out[1007] );

	TF_DeleteTensor( input_tensor );
	TF_DeleteTensor( output_tensor );

	TF_DeleteGraph( graph );

	TF_DeleteStatus( status );

	return 0;
}

int main() {
	return imagenet_classify();
}

TF_Buffer* read_file( const char* file ) {
	FILE *f = fopen( file, "rb" );
	fseek( f, 0, SEEK_END );
	long fsize = ftell( f );
	fseek( f, 0, SEEK_SET );  //same as rewind(f);                                            

	void* data = malloc( fsize );
	fread( data, fsize, 1, f );
	fclose( f );

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	return buf;
}
