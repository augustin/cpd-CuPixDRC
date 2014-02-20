#ifndef SELECTDEVICE_H
#define SELECTDEVICE_H

#include <QDialog>

namespace Ui {
class SelectDevice;
}

class SelectDevice : public QDialog
{
	Q_OBJECT

public:
	explicit SelectDevice(QWidget *parent = 0);
	~SelectDevice();

	int device();

private slots:
    void on_deviceList_itemDoubleClicked();
private:
	Ui::SelectDevice *ui;
	void resizeCols();
};

#endif // SELECTDEVICE_H
