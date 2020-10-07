#include <c10/util/Backtrace.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd {

bool AnomalyMode::_enabled = false;

AnomalyMetadata::~AnomalyMetadata() = default;

void AnomalyMetadata::store_stack() {
  tb_ = c10::get_backtrace();
}

void _print_stack(
    const std::string& stack,
    const std::string& current_node_name,
    bool is_parent) {
  if (stack.empty()) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "No forward pass information available. Enable detect anomaly "
        "during forward pass for more information.");
    return;
  }
  if (!is_parent) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "Traceback of forward call that caused the error:\n",
        stack);
  } else {
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        current_node_name,
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        stack);
  }
}

void AnomalyMetadata::print_stack(const std::string& current_node_name) {
  _print_stack(tb_, current_node_name, false);

  std::shared_ptr<Node>& parent = parent_;
  // if there is no "parent_" in metadata, then it means this metadata's node
  // is the root and stop printing the traceback
  while (parent_) {
    auto pmd = static_cast<AnomalyMetadata*>(parent_->metadata());
    _print_stack(pmd->tb_, parent_->name(), true);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    parent = pmd->parent_;
  }
}

void AnomalyMetadata::assign_parent(const std::shared_ptr<Node>& parent_node) {
  parent_ = parent_node;
}
}}
