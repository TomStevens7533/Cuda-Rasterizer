#define GLEW_STATIC
#include <iostream>
#include <glew.h>
#include <glfw3.h>
#include <glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "obj.h"
#include "ObjLoader.h"
#include "utilities.h"
#include "ext/matrix_clip_space.inl"
#include "CudaKernel.h"


#define FOV_DEG 30
#define MOUSE_SCROLL_SPEED 0.1f

light Light;

//transformations
glm::mat4 glmViewTransform;
glm::mat4 glmProjectionTransform;
glm::mat4 glmMVtransform;
//-------------------------------
//------------Window-------------
//-------------------------------
int height = 800;
int width = 800;
//keyboard control
//mouse control stuff
bool mouseButtonIsDown = false;
float mouseScrollOffset = 0.0f;
double mouseClickedX = 0.0f;
double mouseClickedY = 0.0f;
double rotationX = 0.0f;
double rotationY = 0.0f;
double mouseDeltaX = 0.0f;
double mouseDeltaY = 0.0f;
double deltaX = 0.0f;
double deltaZ = 0.0f;
double cameraMovementIncrement = 0.015f;
//toggle view
bool isFkeyDown = false;
int isFlatShading = false;
bool isMkeyDown = false;
int isMeshView = false;

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

void mainLoop();
void RunCuda();

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);




int main() {

	//Create obj

	std::string data{"obj/cube.obj"};
	mesh = new obj();
	objLoader* loader = new objLoader(data, mesh);
	mesh->buildVBOs();
	std::cout << "Hello world\n";	
	if (InitFramework()) {
		// GLFW main loop
		mainLoop();
	}
	return 0;
}
void mainLoop() {


	while (!glfwWindowShouldClose(window)) {

		//camera rotation, zoom control using mouse
		double* mouseX = new double;
		double* mouseY = new double;

	

		//set up transformations
		float fov_rad = FOV_DEG * PI / 180.0f;
		float AR = width / height;

		//glm::mat4 ModelTransform =utilityCore::buildTransformationMatrix(glm::vec3(0.0f),glm::vec3(0.0f,0.0f,0.0f),glm::vec3(1.0f));
		glm::mat4 ModelTransform = utilityCore::buildTransformationMatrix(glm::vec3(0.0f), glm::vec3(-(rotationY + mouseDeltaY), -(rotationX + mouseDeltaX + 10.0f), 0.0f), glm::vec3(0.6f));

		glm::mat4 cameraAimTransform = utilityCore::buildTransformationMatrix(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f));
		//glm::mat4 cameraAimTransform = utilityCore::buildTransformationMatrix(glm::vec3(0.0f),glm::vec3(-(rotationY + mouseDeltaY),- (rotationX + mouseDeltaX + 10.0f),0.0f),glm::vec3(1.0f));
		glm::mat4 cameraPosTransform = utilityCore::buildTransformationMatrix(glm::vec3(0.0f + deltaX, -.25f + deltaZ, (2.0f + MOUSE_SCROLL_SPEED * mouseScrollOffset)), glm::vec3(0.0f), glm::vec3(1.0f));
		glm::mat4 ViewTransform = cameraAimTransform * cameraPosTransform;
		//glm::mat4 ViewTransform =utilityCore::buildTransformationMatrix(glm::vec3(0.0f + deltaX,-.25f,2.0f + deltaZ + MOUSE_SCROLL_SPEED * mouseScrollOffset),glm::vec3(-(rotationY + mouseDeltaY),- (rotationX + mouseDeltaX),0.0f),glm::vec3(1.0f));

		glmViewTransform = ViewTransform;
		glmProjectionTransform = glm::perspective((float)45.0f, AR, 1.0f, 50.0f);
		glmMVtransform = ViewTransform * ModelTransform;

		//construct light
		Light.position = glm::vec3(7.0f, 2.0f, -10.0f);
		Light.diffColor = glm::vec3(1.0f);
		Light.specColor = glm::vec3(1.0f);
		Light.specExp = 20;
		Light.ambColor = glm::vec3(0.2f, 0.6f, 0.3f);
		glfwPollEvents();

		RunCuda();

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		glfwSwapBuffers(window);
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}
void RunCuda() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr = NULL;

	vbo = mesh->getVBO();
	vbosize = mesh->getVBOsize();

	cbo = mesh->getCBO();
	cbosize = mesh->getCBOsize();

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();

	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, glmViewTransform, glmProjectionTransform, glmMVtransform, Light, isFlatShading, isMeshView);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;
	nbo = NULL;

	frame++;
	fpstracker++;

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
		glfwTerminate();
		return false;
		// Window or OpenGL context creation failed
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);

	//glewExperimental = true;
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		return false;
	
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	// Initialize other stuff
	InitBuffers();
	InitTextures();
	InitCuda();
	InitBuffers();

	return true;
}

void InitCuda() {
	// Use device with highest Gflops/s
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}
void InitBuffers() {

	//Init VBO(FullScreen)
	GLfloat vertices[] =
	{
		-1.0f, -1.0f,
		 1.0f, -1.0f,
		 1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	//PBO Creation
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);
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
