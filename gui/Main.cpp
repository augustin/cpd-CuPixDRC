#include <QApplication>
#include <QDesktopWidget>
#include "DRCwind.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    DRCwind w;
    w.move(app.desktop()->screen()->rect().center() - w.rect().center());
    w.show();

    return app.exec();
}
