#pragma once
#include <stdlib.h>
#include "obj.h"

using namespace std;

class objLoader {
private:
	obj* geomesh;
public:
	objLoader(string, obj*);
	~objLoader();

	//------------------------
	//-------GETTERS----------
	//------------------------

	obj* getMesh();
};

