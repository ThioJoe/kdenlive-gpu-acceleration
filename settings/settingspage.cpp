/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * The module's page in Kdenlive's settings dialog.
 *
 * The module loads this library in Kdenlive's main program and calls
 * gpufx_settings_init() (see src/settings_page_api.h). The page watches for
 * Kdenlive's settings dialog, a KConfigDialog named "settings", and adds
 * itself with KConfigDialog::addPage(). It has its own KCoreConfigSkeleton on
 * the [gpufx] group of Kdenlive's settings file, so the dialog's OK, Apply,
 * Cancel and Defaults buttons handle it like Kdenlive's own pages. When the
 * settings are saved, the module reads them again: the change applies to the
 * monitor at once, and render processes read the file when they start.
 */

#include "settings_page_api.h"

#include <KConfigDialog>
#include <KCoreConfigSkeleton>
#include <KPageWidgetItem>
#include <KSharedConfig>

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDirIterator>
#include <QEvent>
#include <QFile>
#include <QGridLayout>
#include <QGroupBox>
#include <QHash>
#include <QIcon>
#include <QLabel>
#include <QPointer>
#include <QStandardPaths>
#include <QThread>
#include <QVBoxLayout>
#include <QXmlStreamReader>

#include <algorithm>
#include <memory>
#include <vector>

#ifdef GPUFX_SETTINGS_SELFTEST
#include <QAction>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QTimer>
#include <QWindow>

#include <cstdlib>
#endif

#if defined(_WIN32)
#include <stdlib.h>
#include <windows.h>
#define GPUFX_SETTINGS_EXPORT __declspec(dllexport)
#else
#define GPUFX_SETTINGS_EXPORT __attribute__((visibility("default")))
#endif

namespace {

const gpufx_settings_host *g_host = nullptr;

/** Dynamic property of the dialog: its page item, once the page is added */
const char *const kPageProperty = "gpufx_page";

/** Something the module can run on the GPU, and its switch on the page */
struct Service
{
    QString key; // <kind>:<service>, its key in the settings file
    QString kind;
    QString service;
    QString label;
};

/** The names Kdenlive shows for MLT services, from its effect or transition
 *  definitions (several effects can use the same service) */
QHash<QString, QStringList> kdenliveNames(const QString &folder)
{
    QHash<QString, QStringList> names;
    const QStringList dirs = QStandardPaths::locateAll(QStandardPaths::AppDataLocation, folder, QStandardPaths::LocateDirectory);
    for (const QString &dir : dirs) {
        QDirIterator it(dir, {QStringLiteral("*.xml")}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            QFile file(it.next());
            if (!file.open(QIODevice::ReadOnly)) {
                continue;
            }
            QXmlStreamReader xml(&file);
            QString tag;
            while (!xml.atEnd() && xml.readNext() != QXmlStreamReader::Invalid) {
                if (!xml.isStartElement()) {
                    continue;
                }
                if (xml.name() == QLatin1String("effect") || xml.name() == QLatin1String("transition")) {
                    tag = xml.attributes().value(QLatin1String("tag")).toString();
                } else if (xml.name() == QLatin1String("name") && !tag.isEmpty()) {
                    const QString name = xml.readElementText().trimmed();
                    if (!name.isEmpty() && !names[tag].contains(name)) {
                        names[tag].append(name);
                    }
                    break;
                }
            }
        }
    }
    for (QStringList &list : names) {
        list.sort(Qt::CaseInsensitive);
    }
    return names;
}

/** Kdenlive's names, for when its definitions are not found */
QString fallbackName(const QString &key)
{
    static const QHash<QString, QString> names = {
        {QStringLiteral("effect:dropshadow"), QStringLiteral("Drop Shadow")},
        {QStringLiteral("effect:box_blur"), QStringLiteral("BoxBlur")},
        {QStringLiteral("effect:boxblur"), QStringLiteral("Box Blur")},
        {QStringLiteral("effect:brightness"), QStringLiteral("Fade in, Fade out, Intensity (Brightness)")},
        {QStringLiteral("effect:lift_gamma_gain"), QStringLiteral("Lift/gamma/gain")},
        {QStringLiteral("effect:frei0r.squareblur"), QStringLiteral("Square Blur")},
        {QStringLiteral("effect:frei0r.glow"), QStringLiteral("Glow")},
        {QStringLiteral("effect:frei0r.softglow"), QStringLiteral("Soft Glow")},
        {QStringLiteral("effect:frei0r.sharpness"), QStringLiteral("Sharpen")},
        {QStringLiteral("effect:qtblend"), QStringLiteral("Transform")},
        {QStringLiteral("transition:qtblend"), QStringLiteral("Composite and transform")},
    };
    return names.value(key);
}

int kindOrder(const QString &kind)
{
    if (kind == QLatin1String("effect")) {
        return 0;
    }
    return kind == QLatin1String("transition") ? 1 : 2;
}

/** The services of the module, effects first, sorted by name */
const std::vector<Service> &services()
{
    static const std::vector<Service> result = [] {
        const QHash<QString, QStringList> effects = kdenliveNames(QStringLiteral("effects"));
        const QHash<QString, QStringList> transitions = kdenliveNames(QStringLiteral("transitions"));
        std::vector<Service> list;
        for (const QString &key : QString::fromUtf8(g_host->services).split(QLatin1Char(','), Qt::SkipEmptyParts)) {
            Service service;
            service.key = key;
            service.kind = key.section(QLatin1Char(':'), 0, 0);
            service.service = key.section(QLatin1Char(':'), 1);
            if (service.kind == QLatin1String("conversion")) {
                if (service.service == QLatin1String("avformat")) {
                    service.label = QStringLiteral("Video decoding to RGBA");
                } else if (service.service == QLatin1String("avcolor_space")) {
                    service.label = QStringLiteral("Output conversion to YUV");
                }
            } else {
                const QStringList names = (service.kind == QLatin1String("transition") ? transitions : effects).value(service.service);
                service.label = names.isEmpty() ? fallbackName(key) : names.join(QStringLiteral(", "));
                if (key == QLatin1String("transition:qtblend")) {
                    service.label = QStringLiteral("Track compositing (%1)").arg(service.label);
                }
            }
            if (service.label.isEmpty()) {
                service.label = service.service;
            }
            list.push_back(service);
        }
        std::stable_sort(list.begin(), list.end(), [](const Service &a, const Service &b) {
            if (kindOrder(a.kind) != kindOrder(b.kind)) {
                return kindOrder(a.kind) < kindOrder(b.kind);
            }
            return a.kind == QLatin1String("effect") && a.label.compare(b.label, Qt::CaseInsensitive) < 0;
        });
        return list;
    }();
    return result;
}

/** The settings of the page, in the group the module reads */
class Settings : public KCoreConfigSkeleton
{
public:
    explicit Settings(QObject *parent)
        : KCoreConfigSkeleton(KSharedConfig::openConfig(), parent)
        , m_values(new bool[services().size()])
    {
        setCurrentGroup(QString::fromUtf8(g_host->group));
        addItemBool(QStringLiteral("enabled"), m_enabled, true);
        for (size_t i = 0; i < services().size(); ++i) {
            addItemBool(services()[i].key, m_values[i], true);
        }
        read();
    }

    bool enabled() const
    {
        return m_enabled;
    }

    bool value(size_t service) const
    {
        return m_values[service];
    }

private:
    bool m_enabled = true;
    std::unique_ptr<bool[]> m_values;
};

/** Text for the description of the GPU the module gives */
QString gpuStatus(const QString &description)
{
    QHash<QString, QString> values;
    for (const QString &line : description.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) {
        values.insert(line.section(QLatin1Char('='), 0, 0), line.section(QLatin1Char('='), 1));
    }
    if (values.value(QStringLiteral("available")) != QLatin1String("1")) {
        return QStringLiteral("No usable GPU: %1. Everything runs on the CPU.").arg(values.value(QStringLiteral("status")));
    }
    return QStringLiteral("GPU: %1 (compute capability %2, CUDA driver %3, %4 MB)")
        .arg(values.value(QStringLiteral("device")), values.value(QStringLiteral("compute_capability")), values.value(QStringLiteral("cuda_driver")),
             values.value(QStringLiteral("memory_mb")));
}

/** Fills the label with the GPU's description, which can take a moment the
 *  first time CUDA is initialized */
void showGpuStatus(QLabel *label)
{
    QPointer<QLabel> target(label);
    QThread *thread = QThread::create([target]() {
        char text[2048] = {};
        g_host->describe_gpu(text, sizeof(text));
        const QString status = gpuStatus(QString::fromUtf8(text));
        QMetaObject::invokeMethod(
            QCoreApplication::instance(),
            [target, status]() {
                if (target) {
                    target->setText(status);
                }
            },
            Qt::QueuedConnection);
    });
    QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
}

QString pageIcon()
{
    for (const char *name : {"gpu", "cpu", "preferences-system-performance", "video-display"}) {
        if (QIcon::hasThemeIcon(QLatin1String(name))) {
            return QLatin1String(name);
        }
    }
    return QStringLiteral("configure");
}

#ifdef GPUFX_SETTINGS_SELFTEST
/*
 * A variant of the page for the workflow's check in Kdenlive itself, which
 * is not installed: once Kdenlive's main window is up, it opens the settings
 * dialog, shows the page, saves a screenshot of the dialog and a report in
 * the folder GPUFX_SELFTEST_OUTPUT names, and ends Kdenlive.
 */
void selftestFinish(const QString &report, int code)
{
    QFile file(QString::fromLocal8Bit(qgetenv("GPUFX_SELFTEST_OUTPUT")) + QStringLiteral("/kdenlive-report.txt"));
    if (file.open(QIODevice::WriteOnly)) {
        file.write(report.toUtf8());
        file.close();
    }
    std::_Exit(code);
}

/** The application's windows, for a report when the settings dialog does
 *  not open */
QString selftestWindows()
{
    QString text;
    for (QWidget *widget : QApplication::topLevelWidgets()) {
        if (!widget->isVisible() && !qobject_cast<QMainWindow *>(widget)) {
            continue;
        }
        text += QStringLiteral("%1 \"%2\" \"%3\"%4\n")
                    .arg(QLatin1String(widget->metaObject()->className()), widget->objectName(), widget->windowTitle(),
                         widget->isVisible() ? QString() : QStringLiteral(" (hidden)"));
        for (QLabel *label : widget->findChildren<QLabel *>()) {
            if (label->isVisible() && !label->text().isEmpty()) {
                text += QStringLiteral("  %1\n").arg(label->text().left(300));
            }
        }
    }
    if (QWidget *modal = QApplication::activeModalWidget()) {
        text += QStringLiteral("modal: %1 \"%2\"\n").arg(QLatin1String(modal->metaObject()->className()), modal->windowTitle());
    }
    // Qt Quick windows, like Kdenlive's splash screen
    for (QWindow *window : QGuiApplication::topLevelWindows()) {
        if (window->isVisible()) {
            text += QStringLiteral("window: %1 \"%2\"\n").arg(QLatin1String(window->metaObject()->className()), window->title());
        }
    }
    return text;
}

void selftestStart()
{
    auto *timer = new QTimer(QCoreApplication::instance());
    auto clock = std::make_shared<QElapsedTimer>();
    clock->start();
    QObject::connect(timer, &QTimer::timeout, timer, [timer, clock]() {
        for (QWidget *widget : QApplication::topLevelWidgets()) {
            auto *window = qobject_cast<QMainWindow *>(widget);
            QAction *action = window && window->isVisible() ? window->findChild<QAction *>(QStringLiteral("options_configure")) : nullptr;
            if (action) {
                timer->stop();
                action->trigger();
                return;
            }
        }
        if (clock->elapsed() > 150000) {
            selftestFinish(QStringLiteral("Kdenlive's main window did not appear. Windows:\n") + selftestWindows(), 2);
        }
    });
    timer->start(500);
}

void selftestReport(KConfigDialog *dialog, KPageWidgetItem *item, QLabel *status)
{
    auto *timer = new QTimer(dialog);
    auto clock = std::make_shared<QElapsedTimer>();
    clock->start();
    QObject::connect(timer, &QTimer::timeout, dialog, [=]() {
        if (status->text().startsWith(QLatin1String("Checking")) && clock->elapsed() < 60000) {
            return;
        }
        timer->stop();
        dialog->setCurrentPage(item);
        QTimer::singleShot(1000, dialog, [=]() {
            QString report = QStringLiteral("dialog: %1 \"%2\"\n").arg(QLatin1String(dialog->metaObject()->className()), dialog->objectName());
            report += QStringLiteral("current page: %1, icon %2\n").arg(dialog->currentPage() ? dialog->currentPage()->name() : QString(), item->icon().name());
            report += QStringLiteral("status: %1\n").arg(status->text());
            for (QCheckBox *box : item->widget()->findChildren<QCheckBox *>()) {
                report += QStringLiteral("%1: %2%3\n").arg(box->objectName(), box->text(), box->isChecked() ? QString() : QStringLiteral(" (off)"));
            }
            const bool saved = dialog->grab().save(QString::fromLocal8Bit(qgetenv("GPUFX_SELFTEST_OUTPUT")) + QStringLiteral("/kdenlive-settings.png"));
            selftestFinish(report, saved ? 0 : 3);
        });
    });
    timer->start(200);
}
#endif

void addPage(KConfigDialog *dialog)
{
    auto *settings = new Settings(dialog);
    auto *page = new QWidget;
    page->setObjectName(QStringLiteral("gpufx_page"));
    auto *layout = new QVBoxLayout(page);

    // The widgets named kcfg_<key> are handled by the dialog
    auto *enabled = new QCheckBox(QStringLiteral("Use the GPU for the effects and conversions below"), page);
    enabled->setObjectName(QStringLiteral("kcfg_enabled"));
    enabled->setChecked(settings->enabled());
    layout->addWidget(enabled);
    auto *about = new QLabel(QStringLiteral("They run on the NVIDIA GPU when possible, with the same result as on the CPU, and anything the GPU "
                                            "cannot do stays on the CPU. Changes apply to the monitor right away and to renders started afterwards."),
                             page);
    about->setWordWrap(true);
    layout->addWidget(about);
    auto *status = new QLabel(QStringLiteral("Checking the GPU…"), page);
    status->setObjectName(QStringLiteral("gpufx_status"));
    status->setWordWrap(true);
    status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(status);

    // The effects in two columns, read down
    auto *effects = new QGroupBox(QStringLiteral("Effects"), page);
    auto *effectsLayout = new QGridLayout(effects);
    const int effectCount = int(std::count_if(services().begin(), services().end(), [](const Service &service) {
        return service.kind == QLatin1String("effect");
    }));
    const int rows = (effectCount + 1) / 2;
    auto *others = new QGroupBox(QStringLiteral("Compositing and conversions"), page);
    auto *othersLayout = new QVBoxLayout(others);
    int effectIndex = 0;
    for (size_t i = 0; i < services().size(); ++i) {
        const Service &service = services()[i];
        const bool effect = service.kind == QLatin1String("effect");
        auto *box = new QCheckBox(service.label, effect ? effects : others);
        box->setObjectName(QStringLiteral("kcfg_") + service.key);
        box->setToolTip(QStringLiteral("MLT %1: %2").arg(service.kind == QLatin1String("conversion") ? QStringLiteral("service") : service.kind, service.service));
        box->setChecked(settings->value(i));
        if (effect) {
            effectsLayout->addWidget(box, effectIndex % rows, effectIndex / rows);
            ++effectIndex;
        } else {
            othersLayout->addWidget(box);
        }
    }
    layout->addWidget(effects);
    layout->addWidget(others);
    layout->addStretch();
    // The switches of the services only matter with GPU processing on
    effects->setEnabled(settings->enabled());
    others->setEnabled(settings->enabled());
    QObject::connect(enabled, &QCheckBox::toggled, effects, &QWidget::setEnabled);
    QObject::connect(enabled, &QCheckBox::toggled, others, &QWidget::setEnabled);

    KPageWidgetItem *item = dialog->addPage(page, settings, QStringLiteral("GPU Acceleration"), pageIcon(), QStringLiteral("GPU acceleration with NVIDIA CUDA"));
    dialog->setProperty(kPageProperty, QVariant::fromValue<QObject *>(item));
    QObject::connect(settings, &KCoreConfigSkeleton::configChanged, settings, []() {
        g_host->reload_settings();
    });
    showGpuStatus(status);
#ifdef GPUFX_SETTINGS_SELFTEST
    selftestReport(dialog, item, status);
#endif
}

/** Adds the page to Kdenlive's settings dialog when it opens */
class PageInstaller : public QObject
{
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        // Before the dialog's own showEvent(), which loads the pages' values
        if (event->type() == QEvent::Show && watched->isWidgetType()) {
            auto *dialog = qobject_cast<KConfigDialog *>(watched);
            if (dialog && dialog->objectName() == QLatin1String("settings") && !dialog->property(kPageProperty).isValid()) {
                addPage(dialog);
            }
        }
        return QObject::eventFilter(watched, event);
    }
};

/** Render processes started by Kdenlive inherit the path of its settings
 *  file, which depends on how Kdenlive was installed or started */
void exportSettingsFile()
{
    QString file = KSharedConfig::openConfig()->name();
    if (QDir::isRelativePath(file)) {
        file = QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation) + QLatin1Char('/') + file;
    }
    file = QDir::toNativeSeparators(file);
#if defined(_WIN32)
    const auto *path = reinterpret_cast<const wchar_t *>(file.utf16());
    _wputenv_s(L"GPUFX_CONFIG_FILE", path);
    SetEnvironmentVariableW(L"GPUFX_CONFIG_FILE", path);
#else
    qputenv("GPUFX_CONFIG_FILE", QFile::encodeName(file));
#endif
    g_host->reload_settings();
}

void start()
{
    QCoreApplication *app = QCoreApplication::instance();
    if (!qobject_cast<QApplication *>(app)) {
        return;
    }
    exportSettingsFile();
#ifdef GPUFX_SETTINGS_SELFTEST
    selftestStart();
#endif
    // The filter must live in the application's thread
    auto *installer = new PageInstaller;
    installer->moveToThread(app->thread());
    QMetaObject::invokeMethod(
        app,
        [installer, app]() {
            installer->setParent(app);
            app->installEventFilter(installer);
        },
        Qt::AutoConnection);
}

/** Called while the application is being constructed: starts once it is */
void startLater()
{
    QMetaObject::invokeMethod(QCoreApplication::instance(), &start, Qt::QueuedConnection);
}

} // namespace

extern "C" GPUFX_SETTINGS_EXPORT int gpufx_settings_init(const gpufx_settings_host *host)
{
    if (g_host) {
        return 1;
    }
    if (!host || host->version < GPUFX_SETTINGS_API_VERSION) {
        return 0;
    }
    g_host = host;
    if (QCoreApplication::instance()) {
        start();
    } else {
        // Kdenlive creates its application before loading MLT, others may not
        qAddPreRoutine(startLater);
    }
    return 1;
}
