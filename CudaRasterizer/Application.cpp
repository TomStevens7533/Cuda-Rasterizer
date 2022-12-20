#include "Application.h"

bool Application::InitFramework()
{
	glfwSetErrorCallback([](int error, const char* description) {
			std::cerr << "GLFW error detected: " << description << std::endl;
		}
	);
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


	
	//Key callback
	glfwSetKeyCallback(pWindow, [](GLFWwindow* window, int Key, int scancode, int action, int mods)
		{
			//Set GLFW to close next frame
			if (Key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, GL_TRUE);
			}
		}
	);

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
	InitBuffers();
	InitializeBuffers(glm::vec2(width, height));

	GLuint passthroughProgram;
	passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void Application::mainLoop()
{
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
	kernelCleanup();
	cleanupCuda();
}


void Application::InitBuffers()
{
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

void Application::InitTextures()
{
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

GLuint Application::initShader()
{
	const char* attribLocations[] = { "Position", "Tex" };
	shaders out;
	compileShader("Passthrough Vertex", passthroughVS.c_str(), GL_VERTEX_SHADER, (GLint&)out.vertex);
	compileShader("Passthrough Fragment", passthroughFS.c_str(), GL_FRAGMENT_SHADER, (GLint&)out.fragment);

	GLint location;
	GLuint program = glCreateProgram();

	for (GLuint i = 0; i < 2; ++i)
	{
		glBindAttribLocation(program, i, attribLocations[i]);
	}
	glAttachShader(program, out.vertex);
	glAttachShader(program, out.fragment);
	glLinkProgram(program);
	GLint linked;

	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		std::cerr << "Program did not link." << std::endl;
	}

	glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}
void Application::compileShader(const char* shaderName, const char* shaderSource, GLenum shaderType, GLint& shaders) {
	GLint s;
	s = glCreateShader(shaderType);

	GLint slen = (unsigned int)std::strlen(shaderSource);
	char* ss = new char[slen + 1];
	std::strcpy(ss, shaderSource);

	const char* css = ss;
	glShaderSource(s, 1, &css, &slen);

	GLint compiled;
	glCompileShader(s);
	glGetShaderiv(s, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		std::cout << shaderName << " did not compile" << std::endl;
	}

	shaders = s;

	delete[] ss;
}
void Application::cleanupCuda()
{
	if (pbo) deletePBO(&pbo);
	if (displayImage) deleteTexture(&displayImage);
}

void Application::deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void Application::deletePBO(GLuint* pbo)
{
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void Application::RunCuda(std::vector<Triangle>& triangleVector)
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	size_t size;
	cudaGraphicsMapResources(1, &m_cudaGraphicsResource);
	cudaError_t x = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, m_cudaGraphicsResource);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, triangleVector.data(), triangleVector.size());
	cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource);


	frame++;
	fpstracker++;
}

void Application::ShowFPS()
{
	double currentTime = glfwGetTime();
	double delta = currentTime - lastTime;
	frame++;
	if (delta >= 1.0) { // If last cout was more than 1 sec ago

		double fps = double(frame) / delta;

		std::stringstream ss;
		ss << "Soy de meigd" << " [" << fps << " FPS]";

		glfwSetWindowTitle(pWindow, ss.str().c_str());

		frame = 0;
		lastTime = currentTime;
	}
}

