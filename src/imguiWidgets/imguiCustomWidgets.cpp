#include "ImguiCustomWidgets.h"

void ImGui::ShowPerfomanceMetrics()
{

    if (!ImGui::Begin("Perfomance parameters"))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}
