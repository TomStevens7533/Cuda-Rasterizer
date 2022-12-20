#pragma once
#include <iostream>
#define GLEW_STATIC
#define FOV_DEG 30
#define MOUSE_SCROLL_SPEED 0.1f

#include <glew.h>
#include <glfw3.h>
#include <glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "CudaKernel.h"

#include "utilities.h"
#include "ext/matrix_clip_space.inl"


class Application
{
public:
	bool InitFramework();
	void mainLoop();
private:
	void InitBuffers();
	void InitTextures();
	GLuint initShader();
	void cleanupCuda();
	void deleteTexture(GLuint* tex);
	void deletePBO(GLuint* pbo);
	void RunCuda(std::vector<Triangle>& triangleVector);
	void ShowFPS();
	void compileShader(const char* shaderName, const char* shaderSource, GLenum shaderType, GLint& shaders);
private:


	//-------------------------------
	//------------Window-------------
	//-------------------------------
	int height = 800;
	int width = 800;
	int frame{};
	double lastTime{};
	int fpstracker;
	double seconds;
	int fps = 0;
	GLuint positionLocation = 0;
	GLuint texcoordsLocation = 1;
	GLuint pbo = (GLuint)NULL;
	GLuint displayImage;
	uchar4* dptr;
	GLFWwindow* pWindow;
	cudaGraphicsResource_t m_cudaGraphicsResource;
	cudaArray* m_cudaArray;
	cudaTextureObject_t m_texture;

};
static std::string passthroughVS =
"	attribute vec4 Position; \n"
"	attribute vec2 Texcoords; \n"
"	varying vec2 v_Texcoords; \n"
"	\n"
"	void main(void){ \n"
"		v_Texcoords = Texcoords; \n"
"		gl_Position = Position; \n"
"	}";
static std::string passthroughFS =
"	varying vec2 v_Texcoords; \n"
"	\n"
"	uniform sampler2D u_image; \n"
"	\n"
"	void main(void){ \n"
"		gl_FragColor = texture2D(u_image, v_Texcoords); \n"
"	}";

struct shaders
{
	GLuint vertex;
	GLuint fragment;
	GLint geometry;
};