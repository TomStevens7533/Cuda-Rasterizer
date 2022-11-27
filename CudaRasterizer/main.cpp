#define GLEW_STATIC
#include <iostream>
#include <glew.h>
#include <glfw3.h>
#include <glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "obj.h"
#include "ObjLoader.h"

//-------------------------------
//------------Window-------------
//-------------------------------
int height = 800;
int width = 800;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char* attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4* dptr;

GLFWwindow* window;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo;
int nbosize;
//-------------------------------
//----------SETUP----------------
//-------------------------------
bool InitFramework();
void InitCuda();
void InitBuffers();
void InitTextures();

void cleanupCuda();
void deleteTexture(GLuint* tex);
void deletePBO(GLuint* pbo);


//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);




int main() {

	//Create obj

	std::string data{"obj/cube.obj"};
	mesh = new obj();
	objLoader* loader = new objLoader(data, mesh);
	mesh->buildVBOs();
	InitFramework();
	std::cout << "Hello world\n";	
}
//Init Functions
bool InitFramework()
{
	glfwSetErrorCallback(errorCallback);
	if (!glfwInit())
	{
		// Initialization failed
		std::cout << "glfw failed initialization!\n";
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	window = glfwCreateWindow(width, height, "Horsecock ofcourse cock", NULL, NULL);
	if (!window)
	{
		// Window or OpenGL context creation failed
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);

	//glewExperimental = true;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
	

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	// Initialize other stuff
	InitBuffers();
	InitTextures();
	InitCuda();
	InitBuffers();

	while (!glfwWindowShouldClose(window))
	{

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	return true;
}

void InitCuda() {
	// Use device with highest Gflops/s
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}
void InitBuffers() {

}
void InitTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

//Clean
void cleanupCuda() {
	if (pbo) deletePBO(&pbo);
	if (displayImage) deleteTexture(&displayImage);
}
void deletePBO(GLuint* ppbo) {
	if (ppbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*ppbo);

		glBindBuffer(GL_ARRAY_BUFFER, *ppbo);
		glDeleteBuffers(1, ppbo);

		*ppbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}


//GLFW Callbacks
void keyCallback(GLFWwindow* pwindow, int key, int, int action, int)
{
	//Set GLFW to close next frame
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(pwindow, GL_TRUE);
	}
}

void errorCallback(int, const char* description)
{
	std::cout << description << std::endl;
}
void mainLoop()
{

}