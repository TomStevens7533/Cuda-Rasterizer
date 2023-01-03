#pragma once
#include "Window.h"
#include "ChunkBuilder.h"

struct GLFWwindow;
class Application final {
public:
	Application(int width, int height);
	~Application();

	inline static Application& Get() { return *s_Instance; }
	inline Window& GetWindow() { return m_Window; }
	void Start();
	void Update();
private:
	static Application* s_Instance;
	Window m_Window;
	ChunkMesh chk;

};