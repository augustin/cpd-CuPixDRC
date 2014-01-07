#ifndef ERRORLIST_H
#define ERRORLIST_H

#include <QDockWidget>

namespace Ui {
class ErrorList;
}

class ErrorList : public QDockWidget
{
	Q_OBJECT

public:
	explicit ErrorList(QWidget *parent = 0);
	~ErrorList();

    void addItem(QString msg);

private:
	Ui::ErrorList *ui;
    void resizeCols();
};

#endif // ERRORLIST_H
