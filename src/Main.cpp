#include <QApplication>
#include "DRCwind.h"

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);
	DRCwind w;
	w.show();
	
	return app.exec();
}
