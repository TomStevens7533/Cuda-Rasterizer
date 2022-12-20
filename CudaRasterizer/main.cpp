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
#include "InteropFrameBuffer.h"
#include "glslutility.h"


#define FOV_DEG 30
#define MOUSE_SCROLL_SPEED 0.1f

//-------------------------------
//------------Window-------------
//-------------------------------
int height = 800;
int width = 800;
//keyboard control
//mouse control stuff
//toggle view
bool isFkeyDown = false;
int isFlatShading = false;
bool isMkeyDown = false;
int isMeshView = false;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame{};
double lastTime{};
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char* attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4* dptr;

GLFWwindow* pWindow;

//-------------------------------
//----------SETUP----------------
//-------------------------------
bool InitFramework();
void InitCuda();
void InitBuffers();
void InitTextures();
GLuint initShader();

void cleanupCuda();
void deleteTexture(GLuint* tex);
void deletePBO(GLuint* pbo);

void mainLoop();
void RunCuda(std::vector<Triangle>& triangleVector);
void ShowFPS();

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);


cudaGraphicsResource_t m_cudaGraphicsResource;
cudaArray* m_cudaArray;
cudaTextureObject_t m_texture;


int main() {

	//Create obj
	//std::string data{"obj/cube.obj"};
	//auto mesh = new obj();
	//objLoader* loader = new objLoader(data, mesh);
	//mesh->buildVBOs();


	if (InitFramework()) {
		// GLFW main loop
		mainLoop();
	}
	return 0;
}
void mainLoop() {


	while (!glfwWindowShouldClose(pWindow)) {

		//camera rotation, zoom control using mouse
		double* mouseX = new double;
		double* mouseY = new double;

	

		//set up transformations
		float fov_rad = FOV_DEG * PI / 180.0f;
		float AR = width / height;



		std::vector<Triangle> triangleVec;
		Triangle tr;
		tr.vertices[0] = glm::vec3{ 0.f, 0.5f,-1.f };
		tr.vertices[1] = glm::vec3{ -.5f, -0.5f,-1.f };
		tr.vertices[2] = glm::vec3{ 0.5f, -0.5f,-1.f };
		triangleVec.push_back(tr);

		RunCuda(triangleVec);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		glfwSwapBuffers(pWindow);
		glfwPollEvents();
		ShowFPS();
	}

	glfwDestroyWindow(pWindow);
	glfwTerminate();
}
void ShowFPS()
{
	double currentTime = glfwGetTime();
	double delta = currentTime - lastTime;
	frame++;
	if (delta >= 1.0) { // If last cout was more than 1 sec ago
		cout << 1000.0 / double(frame) << endl;

		double fps = double(frame) / delta;

		std::stringstream ss;
		ss << "Soy de meigd" << " [" << fps << " FPS]";

		glfwSetWindowTitle(pWindow, ss.str().c_str());

		frame = 0;
		lastTime = currentTime;
	}
}


void RunCuda(std::vector<Triangle>& triangleVector) {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	size_t size;
	cudaGraphicsMapResources(1, &m_cudaGraphicsResource);
	cudaError_t x = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, m_cudaGraphicsResource);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, triangleVector.data(), triangleVector.size(), glm::mat4{}, glm::mat4{}, glm::mat4{});
	cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource);


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
	pWindow = glfwCreateWindow(width, height, "Very cool rasterizer ow ye", NULL, NULL);
	if (!pWindow)
	{
		glfwTerminate();
		return false;
		// Window or OpenGL context creation failed
	}
	glfwMakeContextCurrent(pWindow);
	glfwSetKeyCallback(pWindow, keyCallback);

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

	GLuint passthroughProgram;
	passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	//glfwSetScrollCallback(window, scroll_callback);

	return true;
}

GLuint initShader() {
	const char* attribLocations[] = { "Position", "Tex" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}
void InitCuda() {
	// Use device with highest Gflops/s deprecated
	//cudaError_t error =  cudaGLSetGLDevice(0);

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
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, pbo, cudaGraphicsRegisterFlagsNone);

}
void InitTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
	//delete?
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
	std::cerr << "GLFW error detected: " << description << std::endl;
}
