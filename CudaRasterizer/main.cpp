#include "Application.h"

int main() {

	//Create obj
	//std::string data{"obj/cube.obj"};
	//auto mesh = new obj();
	//objLoader* loader = new objLoader(data, mesh);
	//mesh->buildVBOs();
	Application* app = new Application();
	if (app->InitFramework()) {
		app->mainLoop();
	}
	else
		std::cerr << "Init of framework failed/n";

	delete app;
	return 0;
}
