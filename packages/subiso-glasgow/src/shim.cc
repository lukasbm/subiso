// pybind11 shim around the Glasgow subgraph solver (MIT, ciaranm/glasgow-subgraph-solver).
//
// The shim takes positional integer-indexed graphs, because that is exactly
// what the C++ InputGraph holds; the Python wrapper in subiso_glasgow maps
// NetworkX node ids to those indices and back.
//
// Besides the raw one-shot entry points there is PreparedGraph: a host
// converted once in C++ and reused across queries, which removes the
// per-query graph-construction cost from amortized workloads.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gss/formats/input_graph.hh>
#include <gss/homomorphism.hh>
#include <gss/restarts.hh>
#include <gss/timeout.hh>
#include <gss/vertex_to_vertex_mapping.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using gss::VertexToVertexMapping;

namespace
{
    using Edges = std::vector<std::pair<int, int>>;
    using Labels = std::vector<std::optional<std::string>>;

    auto build_graph(int n, const Edges & edges, const Labels & labels) -> InputGraph
    {
        const bool has_labels = std::any_of(labels.begin(), labels.end(), [](const auto & label) { return label.has_value(); });
        InputGraph graph(n, has_labels, false);
        for (const auto & [from, to] : edges)
            graph.add_edge(from, to);
        if (has_labels)
        {
            for (int vertex = 0; vertex < n; ++vertex)
                graph.set_vertex_label(vertex, labels[vertex].value_or(""));
        }
        return graph;
    }

    auto make_params(bool induced, double timeout_seconds, bool count) -> gss::HomomorphismParams
    {
        gss::HomomorphismParams params;
        params.induced = induced;
        params.count_solutions = count;
        params.start_time = std::chrono::steady_clock::now();
        // n_threads stays 1, so a no-restarts schedule is the matched default.
        params.restarts_schedule = std::make_unique<gss::NoRestartsSchedule>();
        // The searcher dereferences params.timeout unconditionally: 0s means
        // "no limit", exactly as in Glasgow's own CLI.
        const auto limit = timeout_seconds > 0.0
                               ? std::chrono::seconds(static_cast<long>(std::ceil(timeout_seconds)))
                               : std::chrono::seconds(0);
        params.timeout = std::make_shared<gss::Timeout>(limit);
        return params;
    }

    auto mapping_to_dict(const VertexToVertexMapping & mapping) -> py::dict
    {
        py::dict result;
        for (const auto & [pattern_vertex, target_vertex] : mapping)
            result[py::int_(pattern_vertex)] = py::int_(target_vertex);
        return result;
    }

    auto loooong_to_py(const gss::loooong & value) -> py::int_
    {
        std::ostringstream out;
        out << value;
        return py::int_(py::module_::import("builtins").attr("int")(out.str()));
    }

    auto find_mapping_impl(const InputGraph & pattern, const InputGraph & target, bool induced, double timeout) -> py::object
    {
        auto params = make_params(induced, timeout, false);
        const auto result = gss::solve_homomorphism_problem(pattern, target, params);
        if (result.mapping.empty())
            return py::none();
        return mapping_to_dict(result.mapping);
    }

    auto count_embeddings_impl(const InputGraph & pattern, const InputGraph & target, bool induced, double timeout) -> py::int_
    {
        auto params = make_params(induced, timeout, true);
        const auto result = gss::solve_homomorphism_problem(pattern, target, params);
        return loooong_to_py(result.solution_count);
    }

    auto count_images_impl(const InputGraph & pattern, const InputGraph & target, bool induced, double timeout, long long limit) -> py::int_
    {
        auto params = make_params(induced, timeout, true);
        // Count distinct vertex sets (images), not embeddings: hosting sets do
        // not share a stabiliser once injectivity is dropped.
        auto seen = std::make_shared<std::set<std::vector<int>>>();
        params.enumerate_callback = [seen, limit](const VertexToVertexMapping & mapping) -> bool
        {
            std::vector<int> image;
            image.reserve(mapping.size());
            for (const auto & [pattern_vertex, target_vertex] : mapping)
                image.push_back(target_vertex);
            std::sort(image.begin(), image.end());
            seen->insert(std::move(image));
            return !(limit > 0 && static_cast<long long>(seen->size()) >= limit);
        };
        gss::solve_homomorphism_problem(pattern, target, params);
        return py::int_(seen->size());
    }

    auto enumerate_impl(const InputGraph & pattern, const InputGraph & target, bool induced, double timeout, long long limit) -> py::list
    {
        auto params = make_params(induced, timeout, true);
        auto found = std::make_shared<std::vector<VertexToVertexMapping>>();
        params.enumerate_callback = [found, limit](const VertexToVertexMapping & mapping) -> bool
        {
            found->push_back(mapping);
            // Returning false stops the search right after the limit-th mapping.
            return !(limit > 0 && static_cast<long long>(found->size()) >= limit);
        };
        gss::solve_homomorphism_problem(pattern, target, params);
        py::list result;
        for (const auto & mapping : *found)
            result.append(mapping_to_dict(mapping));
        return result;
    }
}

class PreparedGraph
{
public:
    InputGraph graph;

    explicit PreparedGraph(InputGraph value) : graph(std::move(value)) {}
};

PYBIND11_MODULE(_glasgow, m)
{
    m.doc() = "pybind11 shim around the Glasgow subgraph solver; used by subiso_glasgow.";

    py::class_<PreparedGraph>(m, "PreparedGraph")
        .def_property_readonly(
            "num_vertices", [](const PreparedGraph & prepared) { return prepared.graph.size(); });

    m.def(
        "prepare_graph",
        [](int n, const Edges & edges, const Labels & labels) { return PreparedGraph(build_graph(n, edges, labels)); },
        py::arg("n"), py::arg("edges"), py::arg("labels"));

    m.def(
        "find_mapping_prepared",
        [](const PreparedGraph & target, int pattern_n, const Edges & pattern_edges, const Labels & pattern_labels,
           bool induced, double timeout_seconds) -> py::object
        {
            auto pattern = build_graph(pattern_n, pattern_edges, pattern_labels);
            return find_mapping_impl(pattern, target.graph, induced, timeout_seconds);
        },
        py::arg("target"), py::arg("pattern_n"), py::arg("pattern_edges"), py::arg("pattern_labels"),
        py::arg("induced"), py::arg("timeout_seconds") = 0.0);

    m.def(
        "count_embeddings_prepared",
        [](const PreparedGraph & target, int pattern_n, const Edges & pattern_edges, const Labels & pattern_labels,
           bool induced, double timeout_seconds) -> py::int_
        {
            auto pattern = build_graph(pattern_n, pattern_edges, pattern_labels);
            return count_embeddings_impl(pattern, target.graph, induced, timeout_seconds);
        },
        py::arg("target"), py::arg("pattern_n"), py::arg("pattern_edges"), py::arg("pattern_labels"),
        py::arg("induced"), py::arg("timeout_seconds") = 0.0);

    m.def(
        "count_images_prepared",
        [](const PreparedGraph & target, int pattern_n, const Edges & pattern_edges, const Labels & pattern_labels,
           bool induced, double timeout_seconds, long long limit) -> py::int_
        {
            auto pattern = build_graph(pattern_n, pattern_edges, pattern_labels);
            return count_images_impl(pattern, target.graph, induced, timeout_seconds, limit);
        },
        py::arg("target"), py::arg("pattern_n"), py::arg("pattern_edges"), py::arg("pattern_labels"),
        py::arg("induced"), py::arg("timeout_seconds") = 0.0, py::arg("limit") = 0);

    m.def(
        "enumerate_mappings_prepared",
        [](const PreparedGraph & target, int pattern_n, const Edges & pattern_edges, const Labels & pattern_labels,
           bool induced, double timeout_seconds, long long limit) -> py::list
        {
            auto pattern = build_graph(pattern_n, pattern_edges, pattern_labels);
            return enumerate_impl(pattern, target.graph, induced, timeout_seconds, limit);
        },
        py::arg("target"), py::arg("pattern_n"), py::arg("pattern_edges"), py::arg("pattern_labels"),
        py::arg("induced"), py::arg("timeout_seconds") = 0.0, py::arg("limit") = 0);
}
