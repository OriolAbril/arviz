(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("60b265de-929a-425f-90b6-5f7c31eef7fc");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '60b265de-929a-425f-90b6-5f7c31eef7fc' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"107d5b6d-e877-40cb-b48f-7c7cfbf9144b":{"roots":{"references":[{"attributes":{},"id":"26662","type":"BasicTicker"},{"attributes":{"data_source":{"id":"26756"},"glyph":{"id":"26757"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26758"},"selection_glyph":null,"view":{"id":"26760"}},"id":"26759","type":"GlyphRenderer"},{"attributes":{},"id":"26779","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26723","type":"VBar"},{"attributes":{},"id":"26693","type":"LinearScale"},{"attributes":{"axis":{"id":"26661"},"ticker":null},"id":"26664","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26712","type":"PolyAnnotation"},{"attributes":{},"id":"26706","type":"WheelZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26703"},{"id":"26704"},{"id":"26705"},{"id":"26706"},{"id":"26707"},{"id":"26708"},{"id":"26709"},{"id":"26710"}]},"id":"26713","type":"Toolbar"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26727","type":"Span"},{"attributes":{},"id":"26708","type":"UndoTool"},{"attributes":{},"id":"26800","type":"UnionRenderers"},{"attributes":{},"id":"26709","type":"SaveTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26678","type":"PolyAnnotation"},{"attributes":{},"id":"26801","type":"Selection"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26763","type":"VBar"},{"attributes":{},"id":"26802","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"26712"}},"id":"26707","type":"LassoSelectTool"},{"attributes":{"callback":null},"id":"26710","type":"HoverTool"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26755","type":"Span"},{"attributes":{"data_source":{"id":"26750"},"glyph":{"id":"26751"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26752"},"selection_glyph":null,"view":{"id":"26754"}},"id":"26753","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26665"},"dimension":1,"ticker":null},"id":"26668","type":"Grid"},{"attributes":{},"id":"26798","type":"UnionRenderers"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26758","type":"VBar"},{"attributes":{},"id":"26797","type":"Selection"},{"attributes":{},"id":"26672","type":"WheelZoomTool"},{"attributes":{"text":"tau"},"id":"26748","type":"Title"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26782"},"selection_policy":{"id":"26783"}},"id":"26722","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26752","type":"VBar"},{"attributes":{"data_source":{"id":"26768"},"glyph":{"id":"26769"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26770"},"selection_glyph":null,"view":{"id":"26772"}},"id":"26771","type":"GlyphRenderer"},{"attributes":{},"id":"26659","type":"LinearScale"},{"attributes":{},"id":"26657","type":"LinearScale"},{"attributes":{},"id":"26796","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26779"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26662"}},"id":"26661","type":"LinearAxis"},{"attributes":{"children":[{"id":"26810"},{"id":"26808"}]},"id":"26811","type":"Column"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26742","type":"VBar"},{"attributes":{},"id":"26786","type":"Selection"},{"attributes":{"toolbars":[{"id":"26679"},{"id":"26713"}],"tools":[{"id":"26669"},{"id":"26670"},{"id":"26671"},{"id":"26672"},{"id":"26673"},{"id":"26674"},{"id":"26675"},{"id":"26676"},{"id":"26703"},{"id":"26704"},{"id":"26705"},{"id":"26706"},{"id":"26707"},{"id":"26708"},{"id":"26709"},{"id":"26710"}]},"id":"26809","type":"ProxyToolbar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26784"},"selection_policy":{"id":"26785"}},"id":"26728","type":"ColumnDataSource"},{"attributes":{},"id":"26799","type":"Selection"},{"attributes":{"overlay":{"id":"26677"}},"id":"26671","type":"BoxZoomTool"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26741","type":"VBar"},{"attributes":{},"id":"26787","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26773","type":"Span"},{"attributes":{},"id":"26669","type":"ResetTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26745","type":"Span"},{"attributes":{"children":[[{"id":"26652"},0,0],[{"id":"26688"},0,1]]},"id":"26808","type":"GridBox"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26736","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26770","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26788"},"selection_policy":{"id":"26789"}},"id":"26740","type":"ColumnDataSource"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26799"},"selection_policy":{"id":"26800"}},"id":"26756","type":"ColumnDataSource"},{"attributes":{},"id":"26696","type":"BasicTicker"},{"attributes":{"overlay":{"id":"26711"}},"id":"26705","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"26734"},"glyph":{"id":"26735"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26736"},"selection_glyph":null,"view":{"id":"26738"}},"id":"26737","type":"GlyphRenderer"},{"attributes":{},"id":"26675","type":"SaveTool"},{"attributes":{"axis":{"id":"26699"},"dimension":1,"ticker":null},"id":"26702","type":"Grid"},{"attributes":{"ticks":[0,1,2,3]},"id":"26746","type":"FixedTicker"},{"attributes":{"text":"mu"},"id":"26776","type":"Title"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26711","type":"BoxAnnotation"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26797"},"selection_policy":{"id":"26798"}},"id":"26750","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"26762"},"glyph":{"id":"26763"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26764"},"selection_glyph":null,"view":{"id":"26766"}},"id":"26765","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26794"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26696"}},"id":"26695","type":"LinearAxis"},{"attributes":{},"id":"26789","type":"UnionRenderers"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26796"},"ticker":{"id":"26774"}},"id":"26699","type":"LinearAxis"},{"attributes":{"source":{"id":"26750"}},"id":"26754","type":"CDSView"},{"attributes":{},"id":"26803","type":"Selection"},{"attributes":{},"id":"26703","type":"ResetTool"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26761","type":"Span"},{"attributes":{"axis":{"id":"26695"},"ticker":null},"id":"26698","type":"Grid"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26751","type":"VBar"},{"attributes":{},"id":"26784","type":"Selection"},{"attributes":{},"id":"26691","type":"LinearScale"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26729","type":"VBar"},{"attributes":{"below":[{"id":"26661"}],"center":[{"id":"26664"},{"id":"26668"},{"id":"26727"},{"id":"26733"},{"id":"26739"},{"id":"26745"}],"left":[{"id":"26665"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26725"},{"id":"26731"},{"id":"26737"},{"id":"26743"}],"title":{"id":"26748"},"toolbar":{"id":"26679"},"toolbar_location":null,"x_range":{"id":"26653"},"x_scale":{"id":"26657"},"y_range":{"id":"26655"},"y_scale":{"id":"26659"}},"id":"26652","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"26670","type":"PanTool"},{"attributes":{"source":{"id":"26768"}},"id":"26772","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26724","type":"VBar"},{"attributes":{},"id":"26785","type":"UnionRenderers"},{"attributes":{},"id":"26788","type":"Selection"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26781"},"ticker":{"id":"26746"}},"id":"26665","type":"LinearAxis"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26803"},"selection_policy":{"id":"26804"}},"id":"26768","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26740"}},"id":"26744","type":"CDSView"},{"attributes":{"ticks":[0,1,2,3]},"id":"26774","type":"FixedTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26669"},{"id":"26670"},{"id":"26671"},{"id":"26672"},{"id":"26673"},{"id":"26674"},{"id":"26675"},{"id":"26676"}]},"id":"26679","type":"Toolbar"},{"attributes":{"data_source":{"id":"26722"},"glyph":{"id":"26723"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26724"},"selection_glyph":null,"view":{"id":"26726"}},"id":"26725","type":"GlyphRenderer"},{"attributes":{},"id":"26653","type":"DataRange1d"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26757","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26801"},"selection_policy":{"id":"26802"}},"id":"26762","type":"ColumnDataSource"},{"attributes":{"callback":null},"id":"26676","type":"HoverTool"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26730","type":"VBar"},{"attributes":{"overlay":{"id":"26678"}},"id":"26673","type":"LassoSelectTool"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26769","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26733","type":"Span"},{"attributes":{"source":{"id":"26722"}},"id":"26726","type":"CDSView"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26767","type":"Span"},{"attributes":{},"id":"26704","type":"PanTool"},{"attributes":{"source":{"id":"26762"}},"id":"26766","type":"CDSView"},{"attributes":{},"id":"26674","type":"UndoTool"},{"attributes":{},"id":"26781","type":"BasicTickFormatter"},{"attributes":{},"id":"26782","type":"Selection"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26764","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26786"},"selection_policy":{"id":"26787"}},"id":"26734","type":"ColumnDataSource"},{"attributes":{},"id":"26783","type":"UnionRenderers"},{"attributes":{},"id":"26655","type":"DataRange1d"},{"attributes":{"data_source":{"id":"26728"},"glyph":{"id":"26729"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26730"},"selection_glyph":null,"view":{"id":"26732"}},"id":"26731","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"26740"},"glyph":{"id":"26741"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26742"},"selection_glyph":null,"view":{"id":"26744"}},"id":"26743","type":"GlyphRenderer"},{"attributes":{},"id":"26794","type":"BasicTickFormatter"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26739","type":"Span"},{"attributes":{"source":{"id":"26734"}},"id":"26738","type":"CDSView"},{"attributes":{"source":{"id":"26728"}},"id":"26732","type":"CDSView"},{"attributes":{"toolbar":{"id":"26809"},"toolbar_location":"above"},"id":"26810","type":"ToolbarBox"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26735","type":"VBar"},{"attributes":{"below":[{"id":"26695"}],"center":[{"id":"26698"},{"id":"26702"},{"id":"26755"},{"id":"26761"},{"id":"26767"},{"id":"26773"}],"left":[{"id":"26699"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26753"},{"id":"26759"},{"id":"26765"},{"id":"26771"}],"title":{"id":"26776"},"toolbar":{"id":"26713"},"toolbar_location":null,"x_range":{"id":"26653"},"x_scale":{"id":"26691"},"y_range":{"id":"26655"},"y_scale":{"id":"26693"}},"id":"26688","subtype":"Figure","type":"Plot"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26677","type":"BoxAnnotation"},{"attributes":{},"id":"26804","type":"UnionRenderers"},{"attributes":{"source":{"id":"26756"}},"id":"26760","type":"CDSView"}],"root_ids":["26811"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"107d5b6d-e877-40cb-b48f-7c7cfbf9144b","root_ids":["26811"],"roots":{"26811":"60b265de-929a-425f-90b6-5f7c31eef7fc"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();