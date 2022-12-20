#include "Engine.h"

int main() {

	//Create obj
	//std::string data{"obj/cube.obj"};
	//auto mesh = new obj();
	//objLoader* loader = new objLoader(data, mesh);
	//mesh->buildVBOs();
	Engine* app = new Engine();
	if (app->InitFramework()) {
		app->mainLoop();
	}
	else
		std::cerr << "Init of framework failed/n";

	delete app;
	return 0;
}
