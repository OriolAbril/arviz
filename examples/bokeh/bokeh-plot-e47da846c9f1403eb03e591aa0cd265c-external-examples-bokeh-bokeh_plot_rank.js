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
    
      
      
    
      var element = document.getElementById("e364f5b8-9786-4710-a747-fbecf28b723c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'e364f5b8-9786-4710-a747-fbecf28b723c' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
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
                    
                  var docs_json = '{"370d3205-d3f0-420e-aad5-1842a59b9e92":{"roots":{"references":[{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90623","type":"VBar"},{"attributes":{"callback":null},"id":"90591","type":"HoverTool"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90663"},"selection_policy":{"id":"90664"}},"id":"90603","type":"ColumnDataSource"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90669"},"selection_policy":{"id":"90670"}},"id":"90621","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"90592"}},"id":"90586","type":"BoxZoomTool"},{"attributes":{"source":{"id":"90621"}},"id":"90625","type":"CDSView"},{"attributes":{},"id":"90585","type":"PanTool"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90667"},"selection_policy":{"id":"90668"}},"id":"90615","type":"ColumnDataSource"},{"attributes":{},"id":"90584","type":"ResetTool"},{"attributes":{"ticks":[0,1,2,3]},"id":"90655","type":"FixedTicker"},{"attributes":{},"id":"90590","type":"SaveTool"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"90636","type":"Span"},{"attributes":{},"id":"90587","type":"WheelZoomTool"},{"attributes":{},"id":"90665","type":"Selection"},{"attributes":{"data_source":{"id":"90621"},"glyph":{"id":"90622"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90623"},"selection_glyph":null,"view":{"id":"90625"}},"id":"90624","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"90593"}},"id":"90588","type":"LassoSelectTool"},{"attributes":{},"id":"90666","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"90626","type":"Span"},{"attributes":{},"id":"90589","type":"UndoTool"},{"attributes":{"data_source":{"id":"90631"},"glyph":{"id":"90632"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90633"},"selection_glyph":null,"view":{"id":"90635"}},"id":"90634","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90633","type":"VBar"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90632","type":"VBar"},{"attributes":{"overlay":{"id":"90559"}},"id":"90554","type":"LassoSelectTool"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90678"},"selection_policy":{"id":"90679"}},"id":"90631","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90639","type":"VBar"},{"attributes":{},"id":"90551","type":"PanTool"},{"attributes":{"source":{"id":"90631"}},"id":"90635","type":"CDSView"},{"attributes":{"toolbar":{"id":"90690"},"toolbar_location":"above"},"id":"90691","type":"ToolbarBox"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90680"},"selection_policy":{"id":"90681"}},"id":"90637","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"90550"},{"id":"90551"},{"id":"90552"},{"id":"90553"},{"id":"90554"},{"id":"90555"},{"id":"90556"},{"id":"90557"}]},"id":"90560","type":"Toolbar"},{"attributes":{},"id":"90667","type":"Selection"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90638","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"90558","type":"BoxAnnotation"},{"attributes":{},"id":"90668","type":"UnionRenderers"},{"attributes":{},"id":"90543","type":"BasicTicker"},{"attributes":{},"id":"90675","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"90637"}},"id":"90641","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90645","type":"VBar"},{"attributes":{"toolbars":[{"id":"90560"},{"id":"90594"}],"tools":[{"id":"90550"},{"id":"90551"},{"id":"90552"},{"id":"90553"},{"id":"90554"},{"id":"90555"},{"id":"90556"},{"id":"90557"},{"id":"90584"},{"id":"90585"},{"id":"90586"},{"id":"90587"},{"id":"90588"},{"id":"90589"},{"id":"90590"},{"id":"90591"}]},"id":"90690","type":"ProxyToolbar"},{"attributes":{"axis_label":"Chain","formatter":{"id":"90677"},"ticker":{"id":"90655"}},"id":"90580","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"90592","type":"BoxAnnotation"},{"attributes":{},"id":"90677","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90644","type":"VBar"},{"attributes":{},"id":"90556","type":"SaveTool"},{"attributes":{"data_source":{"id":"90637"},"glyph":{"id":"90638"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90639"},"selection_glyph":null,"view":{"id":"90641"}},"id":"90640","type":"GlyphRenderer"},{"attributes":{},"id":"90553","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"90642","type":"Span"},{"attributes":{"children":[{"id":"90691"},{"id":"90689"}]},"id":"90692","type":"Column"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90682"},"selection_policy":{"id":"90683"}},"id":"90643","type":"ColumnDataSource"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"90620","type":"Span"},{"attributes":{"source":{"id":"90643"}},"id":"90647","type":"CDSView"},{"attributes":{},"id":"90678","type":"Selection"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90651","type":"VBar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"90593","type":"PolyAnnotation"},{"attributes":{},"id":"90679","type":"UnionRenderers"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90650","type":"VBar"},{"attributes":{},"id":"90574","type":"LinearScale"},{"attributes":{},"id":"90669","type":"Selection"},{"attributes":{"data_source":{"id":"90643"},"glyph":{"id":"90644"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90645"},"selection_glyph":null,"view":{"id":"90647"}},"id":"90646","type":"GlyphRenderer"},{"attributes":{},"id":"90555","type":"UndoTool"},{"attributes":{"axis":{"id":"90542"},"ticker":null},"id":"90545","type":"Grid"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"90648","type":"Span"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"90559","type":"PolyAnnotation"},{"attributes":{"axis_label":"Chain","formatter":{"id":"90662"},"ticker":{"id":"90627"}},"id":"90546","type":"LinearAxis"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90684"},"selection_policy":{"id":"90685"}},"id":"90649","type":"ColumnDataSource"},{"attributes":{},"id":"90670","type":"UnionRenderers"},{"attributes":{"source":{"id":"90649"}},"id":"90653","type":"CDSView"},{"attributes":{"data_source":{"id":"90649"},"glyph":{"id":"90650"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90651"},"selection_glyph":null,"view":{"id":"90653"}},"id":"90652","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"90654","type":"Span"},{"attributes":{},"id":"90680","type":"Selection"},{"attributes":{},"id":"90681","type":"UnionRenderers"},{"attributes":{"callback":null},"id":"90557","type":"HoverTool"},{"attributes":{},"id":"90536","type":"DataRange1d"},{"attributes":{"below":[{"id":"90576"}],"center":[{"id":"90579"},{"id":"90583"},{"id":"90636"},{"id":"90642"},{"id":"90648"},{"id":"90654"}],"left":[{"id":"90580"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"90634"},{"id":"90640"},{"id":"90646"},{"id":"90652"}],"title":{"id":"90657"},"toolbar":{"id":"90594"},"toolbar_location":null,"x_range":{"id":"90534"},"x_scale":{"id":"90572"},"y_range":{"id":"90536"},"y_scale":{"id":"90574"}},"id":"90569","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"90572","type":"LinearScale"},{"attributes":{},"id":"90534","type":"DataRange1d"},{"attributes":{},"id":"90682","type":"Selection"},{"attributes":{"axis":{"id":"90546"},"dimension":1,"ticker":null},"id":"90549","type":"Grid"},{"attributes":{},"id":"90683","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90604","type":"VBar"},{"attributes":{"overlay":{"id":"90558"}},"id":"90552","type":"BoxZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"90584"},{"id":"90585"},{"id":"90586"},{"id":"90587"},{"id":"90588"},{"id":"90589"},{"id":"90590"},{"id":"90591"}]},"id":"90594","type":"Toolbar"},{"attributes":{"below":[{"id":"90542"}],"center":[{"id":"90545"},{"id":"90549"},{"id":"90608"},{"id":"90614"},{"id":"90620"},{"id":"90626"}],"left":[{"id":"90546"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"90606"},{"id":"90612"},{"id":"90618"},{"id":"90624"}],"title":{"id":"90629"},"toolbar":{"id":"90560"},"toolbar_location":null,"x_range":{"id":"90534"},"x_scale":{"id":"90538"},"y_range":{"id":"90536"},"y_scale":{"id":"90540"}},"id":"90533","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"90540","type":"LinearScale"},{"attributes":{},"id":"90550","type":"ResetTool"},{"attributes":{"text":"tau"},"id":"90629","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90605","type":"VBar"},{"attributes":{},"id":"90538","type":"LinearScale"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"90608","type":"Span"},{"attributes":{"data_source":{"id":"90603"},"glyph":{"id":"90604"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90605"},"selection_glyph":null,"view":{"id":"90607"}},"id":"90606","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"90660"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"90543"}},"id":"90542","type":"LinearAxis"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90611","type":"VBar"},{"attributes":{},"id":"90684","type":"Selection"},{"attributes":{},"id":"90660","type":"BasicTickFormatter"},{"attributes":{"ticks":[0,1,2,3]},"id":"90627","type":"FixedTicker"},{"attributes":{},"id":"90685","type":"UnionRenderers"},{"attributes":{"source":{"id":"90603"}},"id":"90607","type":"CDSView"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"90675"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"90577"}},"id":"90576","type":"LinearAxis"},{"attributes":{"text":"mu"},"id":"90657","type":"Title"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"90665"},"selection_policy":{"id":"90666"}},"id":"90609","type":"ColumnDataSource"},{"attributes":{},"id":"90662","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90610","type":"VBar"},{"attributes":{"source":{"id":"90609"}},"id":"90613","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90617","type":"VBar"},{"attributes":{"children":[[{"id":"90533"},0,0],[{"id":"90569"},0,1]]},"id":"90689","type":"GridBox"},{"attributes":{"data_source":{"id":"90615"},"glyph":{"id":"90616"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90617"},"selection_glyph":null,"view":{"id":"90619"}},"id":"90618","type":"GlyphRenderer"},{"attributes":{},"id":"90663","type":"Selection"},{"attributes":{"axis":{"id":"90576"},"ticker":null},"id":"90579","type":"Grid"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90622","type":"VBar"},{"attributes":{},"id":"90664","type":"UnionRenderers"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"90616","type":"VBar"},{"attributes":{"data_source":{"id":"90609"},"glyph":{"id":"90610"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"90611"},"selection_glyph":null,"view":{"id":"90613"}},"id":"90612","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"90580"},"dimension":1,"ticker":null},"id":"90583","type":"Grid"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"90614","type":"Span"},{"attributes":{},"id":"90577","type":"BasicTicker"},{"attributes":{"source":{"id":"90615"}},"id":"90619","type":"CDSView"}],"root_ids":["90692"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"370d3205-d3f0-420e-aad5-1842a59b9e92","root_ids":["90692"],"roots":{"90692":"e364f5b8-9786-4710-a747-fbecf28b723c"}}];
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