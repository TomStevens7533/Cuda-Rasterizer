#include "Engine.h"
#include "Renderer.h"
#include "DeltaTime.h"


Engine::Engine() : m_Application{width, height}, m_Camera{60}
{

}

bool Engine::InitFramework()
{
	

	
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

void Engine::mainLoop()
{
	m_Application.Start();
	while (m_Running) {

		//Update time
		auto timeInstance = Time::GetInstance();
		timeInstance->Update();

		//camera rotation, zoom control using mouse
		double* mouseX = new double;
		double* mouseY = new double;

		//set up transformations
		float fov_rad = FOV_DEG * PI / 180.0f;
		float AR = width / height;

		size_t size;
		uchar4* dptr;
		cudaGraphicsMapResources(1, &m_cudaGraphicsResource);
		cudaError_t x = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, m_cudaGraphicsResource);

		m_Camera.UpdateCamera();

		ClearImage(glm::vec2{ width, height });
		//Start scene
		SceneData scData;
		scData.PBOpos = dptr;
		scData.pCamera = &m_Camera;
		scData.resolution = glm::vec2{ width, height };
		Renderer::BeginScene(scData);
		
		//Update application
		m_Application.Update();

		//End Scene
		cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource);


		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		m_Application.GetWindow().SwapWindow();

		//return;
	}

	
	kernelCleanup();
	cleanupCuda();
}


void Engine::InitBuffers()
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

void Engine::InitTextures()
{
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

GLuint Engine::initShader()
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
void Engine::compileShader(const char* shaderName, const char* shaderSource, GLenum shaderType, GLint& shaders) {
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
void Engine::cleanupCuda()
{
	if (pbo) deletePBO(&pbo);
	if (displayImage) deleteTexture(&displayImage);
}

void Engine::deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void Engine::deletePBO(GLuint* pbo)
{
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}


