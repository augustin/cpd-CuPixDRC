#ifndef DRCWIND_H
#define DRCWIND_H

#include <QMainWindow>

namespace Ui {
class DRCwind;
}

class DRCwind : public QMainWindow
{
	Q_OBJECT

public:
	explicit DRCwind(QWidget *parent = 0);
	~DRCwind();

private:
	Ui::DRCwind *ui;
};

#endif // DRCWIND_H
